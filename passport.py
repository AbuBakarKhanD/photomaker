import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

# Use only HALF of the CPU cores for the AI engine (rembg reads OMP_NUM_THREADS).
# Letting it take every core is what makes the whole PC crawl during processing.
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import gc
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np

cv2.setNumThreads(2)
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import PIL.ImageTk as ImageTk
import PIL.ImageDraw as ImageDraw

from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *


# ============================================================
#  Settings
# ============================================================

DPI_A4 = 300
A4_W = int((210 / 25.4) * DPI_A4)
A4_H = int((297 / 25.4) * DPI_A4)

DPI_PHOTO = 275
PASS_W = int(1.25 * DPI_PHOTO)   # one passport photo, in pixels
PASS_H = int(1.5 * DPI_PHOTO)

SUPERSAMPLE = 2          # render at 2x then shrink = smoother edges
MAX_WORK_DIM = 1280      # photos bigger than this are shrunk first (speed + memory)

# Auto framing: if the face is at least this fraction of the photo's width,
# the photo is already head-and-shoulders -> keep framing, only remove the
# background. Smaller faces = wide shot -> crop to head & shoulders.
TIGHT_FACE_WIDTH_RATIO = 0.27

# Crop mode proportions
HEAD_HEIGHT_RATIO = 0.68     # head (crown to chin) as a share of photo height
FACE_TO_HEAD_FACTOR = 1.6    # Haar "face" box -> whole head
TOP_MARGIN_RATIO = 0.08      # gap above the head


# ============================================================
#  Face detection
# ============================================================

_CASCADES = {}


def _cascade(name):
    if name not in _CASCADES:
        _CASCADES[name] = cv2.CascadeClassifier(cv2.data.haarcascades + name)
    return _CASCADES[name]


def detect_face(pil_rgb):
    """Largest face as (x, y, w, h) in pil_rgb coordinates, or None."""
    gray = cv2.cvtColor(np.array(pil_rgb.convert("RGB")), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    scale = 1.0
    if max(h, w) > 900:                       # detect on a smaller copy
        scale = 900.0 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    min_side = max(24, int(min(gray.shape) * 0.05))

    # try progressively more forgiving detectors
    for name, neighbors in (("haarcascade_frontalface_default.xml", 5),
                            ("haarcascade_frontalface_alt2.xml", 4),
                            ("haarcascade_frontalface_default.xml", 3)):
        casc = _cascade(name)
        if casc.empty():
            continue
        faces = casc.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=neighbors,
                                      minSize=(min_side, min_side))
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            return (int(x / scale), int(y / scale), int(fw / scale), int(fh / scale))
    return None


# ============================================================
#  Background removal + cleanup
# ============================================================

_SESSIONS = {}


def release_heavy_models():
    """The birefnet models take gigabytes of RAM - free them once a batch is done.
    (Results are remembered, so re-generating the same photos doesn't need them.)"""
    for name in list(_SESSIONS):
        if name.startswith("birefnet"):
            del _SESSIONS[name]
    gc.collect()


def _set_priority(low):
    """Run below normal priority while processing so other programs stay responsive."""
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000 if low else 0x00000020)
    except Exception:
        pass


def segment_person(pil_rgb, model_name, alpha_matting):
    """Runs rembg. Imported lazily; keeps ONE model loaded at a time."""
    import rembg
    if model_name not in _SESSIONS:
        _SESSIONS.clear()          # drop the previous model so RAM doesn't pile up
        gc.collect()
        _SESSIONS[model_name] = rembg.new_session(model_name=model_name)
    return rembg.remove(
        pil_rgb,
        session=_SESSIONS[model_name],
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=5,
    )


SOLID_ALPHA_THRESHOLD = 70   # model confidence above this = "this is the person"


def harden_alpha(alpha):
    """
    On pale clothing the model is unsure and returns semi-transparent patches
    (a "see-through" shoulder). Make the INTERIOR of the silhouette fully solid
    and keep the model's soft values only in a thin outer band (3px).
    """
    solid = (alpha >= SOLID_ALPHA_THRESHOLD).astype(np.uint8)
    interior = cv2.erode(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    out = alpha.copy()
    out[interior > 0] = 255
    return out


def complete_shoulders(alpha, orig_rgb, face):
    """
    If one shoulder is clearly smaller than the other (typical for a white shirt
    on a white wall), mirror the good side across the face's centre line to
    complete it. The new area is painted with the MIRRORED shirt pixels (so it
    has the shirt's tone and shading, not the wall's), and the seam is feathered.
    Only runs when the user switches it on: a person who is simply off-centre
    in the frame also has one narrower side, and that must not be "repaired".
    Returns (alpha, rgb).
    """
    if face is None:
        return alpha, orig_rgb
    h, w = alpha.shape
    fx, fy, fw, fh = face
    axis = fx + fw / 2.0
    y0 = int(min(h - 1, max(0, fy + fh * 1.0)))          # below the chin
    if y0 >= h - 4 or not (0 < axis < w):
        return alpha, orig_rgb

    solid = alpha > 100
    zone = np.zeros_like(solid)
    zone[y0:, :] = True
    mid = int(round(axis))
    left_area = int((solid & zone)[:, :mid].sum())
    right_area = int((solid & zone)[:, mid:].sum())
    small, big = sorted([left_area, right_area])
    if big == 0 or small > 0.88 * big:                   # already balanced
        return alpha, orig_rgb

    # mirror the solid mask about the vertical axis through the face centre
    cols = np.arange(w)
    src = np.round(2 * axis - cols).astype(int)
    ok_cols = (src >= 0) & (src < w)
    mirrored = np.zeros_like(solid)
    mirrored[:, ok_cols] = solid[:, src[ok_cols]]

    side = np.zeros_like(solid)                          # only touch the smaller side
    if left_area < right_area:
        side[:, :mid] = True
    else:
        side[:, mid:] = True
    candidate = mirrored & ~solid & zone & side

    # colour gate: must look like the shirt (median colour of the torso)
    lab = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref = np.median(lab[solid & zone], axis=0)
    close = np.linalg.norm(lab - ref, axis=2) < 45
    added = (candidate & close).astype(np.uint8)
    added = cv2.medianBlur(added * 255, 5) > 0           # remove speckle / pinholes

    # keep only additions that are connected to the existing body
    n, labels = cv2.connectedComponents((added | solid).astype(np.uint8), connectivity=8)
    body_labels = np.unique(labels[solid])
    body_labels = body_labels[body_labels > 0]
    connected = np.isin(labels, body_labels)
    fill = added & connected

    out = alpha.copy()
    out[fill] = 255

    # paint the new area with the mirrored shirt pixels, feathering the seam
    mirrored_rgb = np.zeros_like(orig_rgb)
    mirrored_rgb[:, ok_cols] = orig_rgb[:, src[ok_cols]]
    weight = cv2.GaussianBlur(fill.astype(np.float32), (0, 0), 2.5)[:, :, None]
    weight = np.where(fill[:, :, None], 1.0, weight)
    rgb = orig_rgb.astype(np.float32) * (1 - weight) + mirrored_rgb.astype(np.float32) * weight
    return out, np.clip(rgb, 0, 255).astype(np.uint8)


def build_cutout(orig_rgb, raw_alpha, face=None, repair_shoulders=False):
    """
    Turns the model's mask into a clean RGBA cutout.
    Colors always come from the ORIGINAL photo (never the model's output,
    whose edge pixels are darkened), so there are no dark halos.
    """
    alpha = raw_alpha.copy()
    h, w = alpha.shape

    # 1. keep only the biggest piece (drops floating specks)
    mask = (alpha > 30).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 2:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        alpha[labels != largest] = 0

    # 1b. make see-through patches solid; optionally repair a missing shoulder
    alpha = harden_alpha(alpha)
    if repair_shoulders:
        alpha, orig_rgb = complete_shoulders(alpha, orig_rgb, face)

    # 2. fill genuine holes (e.g. a gap inside a white shirt).
    #    A hole = background NOT connected to ANY edge of the picture.
    #    Padding with a 1px ring of background first makes every edge-touching
    #    background region one connected area, so only truly enclosed pockets
    #    are treated as holes.
    person = (alpha > 30).astype(np.uint8) * 255
    padded = cv2.copyMakeBorder(cv2.bitwise_not(person), 1, 1, 1, 1,
                                cv2.BORDER_CONSTANT, value=255)
    cv2.floodFill(padded, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 128)
    holes = (padded[1:-1, 1:-1] == 255)
    if holes.any():
        hn, hl, hs, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), connectivity=4)
        limit = 0.12 * float((person > 0).sum())      # never fill big areas
        for i in range(1, hn):
            if hs[i, cv2.CC_STAT_AREA] <= limit:
                alpha[hl == i] = 255

    # 3. tighten + soften the outline slightly
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = cv2.erode(alpha, k3)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.9)

    # 4. remove old-background color from edge pixels: repaint them with the
    #    color of the nearby solid interior (works for ANY background color)
    core = cv2.erode((alpha >= 250).astype(np.uint8), k3, iterations=2).astype(np.float32)
    rgb_f = orig_rgb.astype(np.float32)
    num = cv2.GaussianBlur(rgb_f * core[:, :, None], (0, 0), 3)
    den = cv2.GaussianBlur(core, (0, 0), 3)[:, :, None]
    interior = num / np.maximum(den, 1e-4)
    edge = (alpha < 250)[:, :, None] & (den > 0.02)
    rgb_f = np.where(edge, interior, rgb_f)

    return np.dstack([np.clip(rgb_f, 0, 255).astype(np.uint8), alpha])


# ============================================================
#  Framing: tight shot vs wide shot
# ============================================================

def decide_framing(framing, face, size):
    """Returns (mode, note). mode is 'keep' or 'crop'."""
    if face is None:
        return "keep", "face not found - kept original framing"
    if framing == "crop":
        return "crop", "cropped to head & shoulders"
    if framing == "keep":
        return "keep", "kept original framing"
    if face[2] / float(size[0]) >= TIGHT_FACE_WIDTH_RATIO:
        return "keep", "already head & shoulders - background only"
    return "crop", "wide shot - cropped to head & shoulders"


def head_shoulder_window(face, size):
    """Region around the face that contains head + shoulders."""
    fx, fy, fw, fh = face
    W, H = size
    cx = fx + fw / 2.0
    x0 = int(max(0, cx - 2.3 * fw))
    x1 = int(min(W, cx + 2.3 * fw))
    y0 = int(max(0, fy - 1.1 * fh))
    y1 = int(min(H, fy + 3.2 * fh))
    return x0, y0, x1, y1


class PhotoResult:
    """A finished cutout. Background color is applied later, so switching
    white/blue or the number of copies never repeats the slow step."""

    def __init__(self, cutout, face, mode, note):
        self.cutout = cutout
        self.face = face
        self.mode = mode
        self.note = note
        self.seconds = 0.0
        self.cached = False
        alpha = np.array(cutout)[:, :, 3]
        rows = (alpha > 30).sum(axis=1)
        fw = face[2] if face else 20
        idx = np.where(rows >= max(3, int(0.15 * fw)))[0]   # ignores thin spikes
        crown = int(idx[0]) if len(idx) else 0
        if face:
            # A real crown (hair / hijab / cap) sits between 0.15 and 0.85 face-heights
            # above the top of the detected face box. Clamping to that range means a
            # stray blob above the head can only shift the framing a little, instead
            # of dragging the whole photo out of position.
            fy, fh = face[1], face[3]
            crown = int(min(max(crown, fy - 0.85 * fh), fy - 0.15 * fh))
        self.crown_y = crown
        self.bottom_y = int(idx[-1]) if len(idx) else cutout.height - 1


_RESULT_CACHE = OrderedDict()
_CACHE_MAX = 16          # remember the last 16 photos only (keeps RAM bounded)


def process_photo(path, model_name, alpha_matting, framing, repair_shoulders=False):
    key = (path, os.path.getmtime(path), model_name, alpha_matting, framing, repair_shoulders)
    if key in _RESULT_CACHE:
        _RESULT_CACHE.move_to_end(key)
        hit = _RESULT_CACHE[key]
        hit.cached = True
        return hit

    started = time.perf_counter()
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")   # honour phone rotation
    longest = max(img.size)
    if longest > MAX_WORK_DIM:
        s = MAX_WORK_DIM / float(longest)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))),
                         Image.Resampling.LANCZOS)

    face = detect_face(img)
    mode, note = decide_framing(framing, face, img.size)

    if mode == "crop":
        x0, y0, x1, y1 = head_shoulder_window(face, img.size)
        img = img.crop((x0, y0, x1, y1))
        face = (face[0] - x0, face[1] - y0, face[2], face[3])

    raw = segment_person(img, model_name, alpha_matting)
    raw_alpha = np.array(raw.convert("RGBA"))[:, :, 3]
    cut = build_cutout(np.array(img), raw_alpha, face, repair_shoulders)

    result = PhotoResult(Image.fromarray(cut), face, mode, note)
    result.seconds = time.perf_counter() - started
    _RESULT_CACHE[key] = result
    while len(_RESULT_CACHE) > _CACHE_MAX:
        _RESULT_CACHE.popitem(last=False)
    return result


def _place_crop(r, cw, ch):
    fx, fy, fw, fh = r.face
    top_margin = ch * TOP_MARGIN_RATIO
    scale = (ch * HEAD_HEIGHT_RATIO) / (fh * FACE_TO_HEAD_FACTOR)
    avail = r.bottom_y - r.crown_y
    if avail > 0:                                    # body must reach the bottom edge
        scale = max(scale, (ch - top_margin) / avail)
    scale = min(scale, (cw * 0.9) / fw)              # but never make the head wider than the frame
    ox = cw / 2.0 - (fx + fw / 2.0) * scale          # face dead-centre
    oy = top_margin - r.crown_y * scale
    return scale, ox, oy


def _place_keep(r, cw, ch):
    W, H = r.cutout.size
    scale = max(cw / float(W), ch / float(H))        # "cover": the smallest possible crop
    nw, nh = W * scale, H * scale
    face_cx = (r.face[0] + r.face[2] / 2.0) * scale if r.face else nw / 2.0
    ox = min(0.0, max(cw - nw, cw / 2.0 - face_cx))  # centre on face, never leave gaps
    oy = 0.0
    if nh > ch and r.face:                           # trim from the bottom, never the chin
        chin = (r.face[1] + r.face[3] * 1.15) * scale
        if chin > ch:
            oy = -min(nh - ch, chin - ch)
    return scale, ox, oy


def render_photo(result, bg_color):
    """Composites the cutout onto the chosen background -> one passport photo."""
    ss = SUPERSAMPLE
    cw, ch = PASS_W * ss, PASS_H * ss
    cut = result.cutout
    if result.mode == "crop" and result.face is not None:
        scale, ox, oy = _place_crop(result, cw, ch)
    else:
        scale, ox, oy = _place_keep(result, cw, ch)

    new_size = (max(1, int(round(cut.width * scale))), max(1, int(round(cut.height * scale))))
    resized = cut.resize(new_size, Image.Resampling.LANCZOS)
    layer = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    layer.paste(resized, (int(round(ox)), int(round(oy))))
    canvas = Image.alpha_composite(Image.new("RGBA", (cw, ch), tuple(bg_color) + (255,)), layer)
    return canvas.convert("RGB").resize((PASS_W, PASS_H), Image.Resampling.LANCZOS)


# ============================================================
#  Page layout / PDF
# ============================================================

def build_pages(photos, copies, stroke_width=7, stroke_color=(0, 0, 0), draw_stroke=True):
    margin_x, spacing_x, spacing_y, top = 35, 65, 75, 15
    pages = []
    page = Image.new("RGB", (A4_W, A4_H), (255, 255, 255))
    x, y = margin_x, top

    for photo in photos:
        for _ in range(copies):
            if x + PASS_W > A4_W - margin_x:
                x = margin_x
                y += PASS_H + spacing_y
            if y + PASS_H > A4_H - margin_x:
                pages.append(page)
                page = Image.new("RGB", (A4_W, A4_H), (255, 255, 255))
                x, y = margin_x, top
            page.paste(photo, (x, y))
            if draw_stroke:                      # black cut-line: white background only
                ImageDraw.Draw(page).rectangle([x, y, x + PASS_W - 1, y + PASS_H - 1],
                                               outline=stroke_color, width=stroke_width)
            x += PASS_W + spacing_x
    pages.append(page)
    return pages


def save_pages_as_pdf(pages, output_pdf_path):
    pages[0].save(output_pdf_path, save_all=True, append_images=pages[1:],
                  resolution=300.0, quality=100)


# ============================================================
#  GUI
# ============================================================

class PassportApp(ttk.Window):
    FRAMING_CHOICES = {
        "Auto (recommended)": "auto",
        "Always crop to head & shoulders": "crop",
        "Keep original framing (background only)": "keep",
    }
    MODEL_CHOICES = ["u2net", "silueta", "birefnet-portrait", "birefnet-general-lite",
                     "u2net_human_seg", "isnet-general-use"]

    def __init__(self):
        super().__init__(themename="flatly")
        self.title("Passport Photo Maker")
        self.geometry("780x800")
        self.minsize(740, 760)

        self.selected_files = []
        self.thumb_images = []
        self.preview_image = None
        self.dark_mode = False
        self.generated_pages = None

        self._build_header()
        self._build_dropzone()
        self._build_options()
        self._build_action_bar()

    # ---------------- UI ----------------

    def _build_header(self):
        header = ttk.Frame(self, padding=(20, 16))
        header.pack(fill=X)
        ttk.Label(header, text="🪪  Passport Photo Maker",
                  font=("Segoe UI", 20, "bold")).pack(side=LEFT)
        self.theme_btn = ttk.Button(header, text="🌙  Dark Mode", bootstyle="secondary-outline",
                                    command=self._toggle_theme, width=14)
        self.theme_btn.pack(side=RIGHT)

    def _build_dropzone(self):
        card = ttk.Labelframe(self, text="1.  Select Photos", padding=16, bootstyle="primary")
        card.pack(fill=X, padx=20, pady=(0, 14))

        top_row = ttk.Frame(card)
        top_row.pack(fill=X)
        ttk.Button(top_row, text="📂  Browse Photo(s)...", bootstyle="primary",
                   command=self.select_photos).pack(side=LEFT)
        ttk.Button(top_row, text="Clear", bootstyle="secondary-outline",
                   command=self.clear_photos).pack(side=LEFT, padx=(10, 0))
        self.files_label = ttk.Label(top_row, text="No photos selected.", bootstyle="secondary")
        self.files_label.pack(side=LEFT, padx=14)

        thumb_container = ttk.Frame(card)
        thumb_container.pack(fill=X, pady=(12, 0))
        self.thumb_canvas = ttk.Canvas(thumb_container, height=100, highlightthickness=0)
        self.thumb_canvas.pack(side=TOP, fill=X)
        h_scroll = ttk.Scrollbar(thumb_container, orient=HORIZONTAL, command=self.thumb_canvas.xview)
        h_scroll.pack(side=TOP, fill=X)
        self.thumb_canvas.configure(xscrollcommand=h_scroll.set)
        self.thumb_frame = ttk.Frame(self.thumb_canvas)
        self.thumb_canvas.create_window((0, 0), window=self.thumb_frame, anchor="nw")
        self.thumb_frame.bind(
            "<Configure>",
            lambda e: self.thumb_canvas.configure(scrollregion=self.thumb_canvas.bbox("all")))

    def _build_options(self):
        card = ttk.Labelframe(self, text="2.  Options", padding=16, bootstyle="info")
        card.pack(fill=X, padx=20, pady=(0, 14))

        row = ttk.Frame(card)
        row.pack(fill=X)

        bg_col = ttk.Frame(row)
        bg_col.pack(side=LEFT, padx=(0, 40))
        ttk.Label(bg_col, text="Background Color", font=("Segoe UI", 10, "bold")).pack(anchor=W, pady=(0, 6))
        self.bg_var = ttk.StringVar(value="white")
        seg = ttk.Frame(bg_col)
        seg.pack(anchor=W)
        ttk.Radiobutton(seg, text="⚪  White", variable=self.bg_var, value="white",
                        bootstyle="toolbutton").pack(side=LEFT, padx=(0, 6))
        ttk.Radiobutton(seg, text="🔵  Blue", variable=self.bg_var, value="blue",
                        bootstyle="toolbutton").pack(side=LEFT)

        copies_col = ttk.Frame(row)
        copies_col.pack(side=LEFT)
        ttk.Label(copies_col, text="Copies per Photo", font=("Segoe UI", 10, "bold")).pack(anchor=W, pady=(0, 6))
        counter = ttk.Frame(copies_col)
        counter.pack(anchor=W)
        self.copies_var = ttk.IntVar(value=4)
        ttk.Button(counter, text="–", width=3, bootstyle="secondary-outline",
                   command=lambda: self._bump_copies(-1)).pack(side=LEFT)
        ttk.Entry(counter, textvariable=self.copies_var, width=6, justify=CENTER,
                  font=("Segoe UI", 11)).pack(side=LEFT, padx=6)
        ttk.Button(counter, text="+", width=3, bootstyle="secondary-outline",
                   command=lambda: self._bump_copies(1)).pack(side=LEFT)

        self.preview_label = ttk.Label(row, text="Preview appears here\nafter generating",
                                       bootstyle="secondary", anchor=CENTER, justify=CENTER)
        self.preview_label.pack(side=RIGHT, padx=(20, 0))

        framing_row = ttk.Frame(card)
        framing_row.pack(fill=X, pady=(14, 0))
        ttk.Label(framing_row, text="Framing:", font=("Segoe UI", 9), width=18).pack(side=LEFT)
        self.framing_var = ttk.StringVar(value="Auto (recommended)")
        ttk.Combobox(framing_row, textvariable=self.framing_var, state="readonly", width=40,
                     values=list(self.FRAMING_CHOICES.keys())).pack(side=LEFT)

        model_row = ttk.Frame(card)
        model_row.pack(fill=X, pady=(8, 0))
        ttk.Label(model_row, text="Removal model:", font=("Segoe UI", 9), width=18).pack(side=LEFT)
        self.model_var = ttk.StringVar(value="u2net")
        ttk.Combobox(model_row, textvariable=self.model_var, state="readonly", width=40,
                     values=self.MODEL_CHOICES).pack(side=LEFT)

        ttk.Label(card, text="Tip: birefnet models cut best but are large and slow on weaker PCs - use u2net for quick jobs.",
                  bootstyle="secondary", font=("Segoe UI", 8)).pack(anchor=W, pady=(4, 0))

        self.matting_var = ttk.BooleanVar(value=False)
        ttk.Checkbutton(card, text="Smoother hair edges (noticeably slower per photo)",
                        variable=self.matting_var, bootstyle="round-toggle").pack(anchor=W, pady=(10, 0))

        self.repair_var = ttk.BooleanVar(value=False)
        ttk.Checkbutton(card, text="Repair a MISSING shoulder (only if the cut-out lost one - can look fake otherwise)",
                        variable=self.repair_var, bootstyle="round-toggle").pack(anchor=W, pady=(6, 0))

    def _build_action_bar(self):
        card = ttk.Frame(self, padding=16)
        card.pack(fill=X, padx=20, pady=(0, 20))

        self.generate_btn = ttk.Button(card, text="✨  Generate", bootstyle="success",
                                       command=self.on_generate, state=DISABLED, width=16)
        self.generate_btn.pack(side=LEFT)
        self.save_btn = ttk.Button(card, text="💾  Save PDF...", bootstyle="primary-outline",
                                   command=self.on_save, state=DISABLED, width=16)
        self.save_btn.pack(side=LEFT, padx=(10, 0))

        status_col = ttk.Frame(card)
        status_col.pack(side=LEFT, fill=X, expand=True, padx=16)
        self.status_var = ttk.StringVar(value="Select photos to begin.")
        ttk.Label(status_col, textvariable=self.status_var, bootstyle="secondary",
                  wraplength=330, justify=LEFT).pack(anchor=W)
        self.progress = ttk.Progressbar(status_col, bootstyle="success-striped",
                                        length=300, mode="determinate")
        self.progress.pack(fill=X, pady=(6, 0))

    # ---------------- behaviour ----------------

    def _toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.style.theme_use("darkly" if self.dark_mode else "flatly")
        self.theme_btn.config(text="☀️  Light Mode" if self.dark_mode else "🌙  Dark Mode")

    def _bump_copies(self, delta):
        try:
            current = self.copies_var.get()
        except Exception:
            current = 1
        self.copies_var.set(max(1, current + delta))

    def select_photos(self):
        paths = filedialog.askopenfilenames(
            title="Select Photo(s)",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not paths:
            return
        self.selected_files = list(paths)
        self.generated_pages = None
        self.save_btn.config(state=DISABLED)
        self._refresh_files_label()
        self._show_thumbnails()
        self.generate_btn.config(state=NORMAL)

    def clear_photos(self):
        self.selected_files = []
        self.generated_pages = None
        self.save_btn.config(state=DISABLED)
        self._refresh_files_label()
        self._show_thumbnails()
        self.generate_btn.config(state=DISABLED)

    def _refresh_files_label(self):
        n = len(self.selected_files)
        self.files_label.config(text="No photos selected." if n == 0 else f"{n} photo(s) selected.")

    def _show_thumbnails(self):
        for widget in self.thumb_frame.winfo_children():
            widget.destroy()
        self.thumb_images = []
        for path in self.selected_files:
            try:
                img = Image.open(path)
                img.thumbnail((170, 170))            # fast: JPEGs decode at reduced size
                img = ImageOps.exif_transpose(img)
                img.thumbnail((85, 85))
                tk_img = ImageTk.PhotoImage(img)
                self.thumb_images.append(tk_img)
                cell = ttk.Frame(self.thumb_frame, padding=3, bootstyle="light")
                ttk.Label(cell, image=tk_img).pack()
                cell.pack(side=LEFT, padx=4, pady=4)
            except Exception:
                pass

    def on_generate(self):
        """Builds the result in memory only. Nothing is written to disk until
        you click Save PDF, so you can freely try other settings."""
        if not self.selected_files:
            return
        try:
            copies = int(self.copies_var.get())
        except Exception:
            messagebox.showerror("Invalid Input", "Please enter a whole number in the copies field.")
            return
        if copies <= 0:
            messagebox.showerror("Invalid Input", "Copies must be 1 or more.")
            return

        bg_color = (255, 255, 255) if self.bg_var.get() == "white" else (16, 145, 211)
        framing = self.FRAMING_CHOICES.get(self.framing_var.get(), "auto")

        self.generate_btn.config(state=DISABLED)
        self.save_btn.config(state=DISABLED)
        self.generated_pages = None
        self.progress["value"] = 0
        self.progress["maximum"] = len(self.selected_files)
        self.status_var.set("Starting...")

        threading.Thread(
            target=self._run_pipeline,
            args=(list(self.selected_files), bg_color, copies,
                  self.matting_var.get(), self.model_var.get(), framing,
                  self.repair_var.get()),
            daemon=True,
        ).start()

    def _run_pipeline(self, files, bg_color, copies, alpha_matting, model_name, framing, repair):
        _set_priority(True)
        try:
            total = len(files)
            photos, modes, problems = [], [], 0
            for i, path in enumerate(files, start=1):
                name = os.path.basename(path)
                self.after(0, self._update_progress, i - 1, total,
                           f"Photo {i}/{total}: {name} - removing background...")
                result = process_photo(path, model_name, alpha_matting, framing, repair)
                photos.append(render_photo(result, bg_color))
                modes.append(result.mode)
                if "not found" in result.note:
                    problems += 1
                if i == 1:
                    self.after(0, self._update_preview, photos[0])
                took = "cached" if result.cached else f"{result.seconds:.1f}s"
                self.after(0, self._update_progress, i, total, f"Photo {i}/{total}: {name} - {result.note} ({took})")
            pages = build_pages(photos, copies, draw_stroke=(tuple(bg_color) == (255, 255, 255)))
            self.after(0, self._on_generate_success, pages, modes, problems)
        except Exception as e:
            self.after(0, self._on_error, str(e))
        finally:
            _set_priority(False)
            release_heavy_models()

    def _update_progress(self, done, total, text):
        self.progress["value"] = done
        self.status_var.set(text)

    def _update_preview(self, pil_img):
        thumb = pil_img.copy()
        thumb.thumbnail((120, 150))
        self.preview_image = ImageTk.PhotoImage(thumb)
        self.preview_label.config(image=self.preview_image, text="")

    def _on_generate_success(self, pages, modes, problems):
        self.generated_pages = pages
        cropped = modes.count("crop")
        kept = modes.count("keep")
        msg = f"✅ Ready - {len(pages)} page(s). {cropped} cropped, {kept} kept as-is. Click 'Save PDF'."
        if problems:
            msg += f"  ⚠ Face not found in {problems} photo(s) - check them."
        self.status_var.set(msg)
        self.generate_btn.config(state=NORMAL)
        self.save_btn.config(state=NORMAL)

    def on_save(self):
        if not self.generated_pages:
            messagebox.showwarning("Nothing to Save", "Click 'Generate' first.")
            return
        path = filedialog.asksaveasfilename(title="Save Printable PDF", defaultextension=".pdf",
                                            filetypes=[("PDF files", "*.pdf")])
        if not path:
            return
        try:
            save_pages_as_pdf(self.generated_pages, path)
            self.status_var.set("✅ PDF saved.")
            messagebox.showinfo("Success", f"PDF saved at:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _on_error(self, message):
        self.status_var.set("❌ Error occurred.")
        self.generate_btn.config(state=NORMAL)
        messagebox.showerror("Error", message)


if __name__ == "__main__":
    app = PassportApp()
    app.mainloop()