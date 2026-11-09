"""
fft_image_editor.py
Frequency-domain image editing with the Fast Fourier Transform, built on Pillow.

Core idea (convolution theorem):
    filtering in the spatial domain  ==  multiplication in the frequency domain
Every edit follows the same pipeline:

    PIL image -> array -> FFT -> multiply by mask H(u,v) -> IFFT -> PIL image

Only the mask H changes between edits.

Dependencies: numpy, Pillow (tkinter for the optional GUI).

Python:
    from PIL import Image
    from fft_image_editor import FFTEditor

    out = FFTEditor(Image.open("photo.jpg")).sharpen(amount=2).blur(cutoff=0.3).image
    out.save("result.png")

Command line (single file or whole folder, edits are chained in order):
    python fft_image_editor.py photo.jpg out.png -e sharpen:amount=2 -e blur:cutoff=0.3
    python fft_image_editor.py ./photos ./edited -e auto_notch -e sharpen -r -j 4
    python fft_image_editor.py --list          # all edits and their settings
    python fft_image_editor.py --gui           # interactive editor
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


# ==========================================================================
# 1. Core FFT pipeline (pure NumPy, works on 2D float arrays)
# ==========================================================================
def fft2c(channel):
    """2D FFT with the zero frequency shifted to the center."""
    return np.fft.fftshift(np.fft.fft2(channel))


def ifft2c(spectrum):
    """Inverse of fft2c; returns the real part."""
    return np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))


def _freq_axes(shape):
    rows, cols = shape
    v = np.fft.fftshift(np.fft.fftfreq(rows))[:, None]
    u = np.fft.fftshift(np.fft.fftfreq(cols))[None, :]
    return u, v


def _freq_grid(shape):
    """Distance from the spectrum center in cycles/pixel (0.5 = Nyquist)."""
    u, v = _freq_axes(shape)
    return np.sqrt(u ** 2 + v ** 2)


def filter_channel(channel, mask_fn, pad=True):
    """Filter one 2D channel. With pad=True the channel is reflect-padded
    first, so the FFT's implicit wrap-around does not create edge artifacts."""
    channel = np.asarray(channel, dtype=np.float64)
    h, w = channel.shape
    if pad:
        ph, pw = min(h // 2, 64), min(w // 2, 64)
        channel = np.pad(channel, ((ph, ph), (pw, pw)), mode="reflect")
    out = ifft2c(fft2c(channel) * mask_fn(channel.shape))
    if pad:
        out = out[ph:ph + h, pw:pw + w]
    return out


# ==========================================================================
# 2. Mask builders: each returns a function  shape -> H(u,v)
# ==========================================================================
def gaussian_lowpass(cutoff):
    """Blur. cutoff = sigma in cycles/pixel (smaller = stronger blur)."""
    return lambda s: np.exp(-(_freq_grid(s) ** 2) / (2 * cutoff ** 2))


def gaussian_highpass(cutoff):
    low = gaussian_lowpass(cutoff)
    return lambda s: 1.0 - low(s)


def ideal_lowpass(cutoff):
    """Hard cutoff (visible ringing; mostly for demonstration)."""
    return lambda s: (_freq_grid(s) <= cutoff).astype(np.float64)


def butterworth_lowpass(cutoff, order=2):
    return lambda s: 1.0 / (1.0 + (_freq_grid(s) / cutoff) ** (2 * order))


def butterworth_highpass(cutoff, order=2):
    low = butterworth_lowpass(cutoff, order)
    return lambda s: 1.0 - low(s)


def bandpass(low_cut, high_cut):
    """Keep only frequencies between low_cut and high_cut."""
    hp, lp = gaussian_highpass(low_cut), gaussian_lowpass(high_cut)
    return lambda s: hp(s) * lp(s)


def notch_reject(centers, radius=0.01):
    """Remove periodic noise. `centers` = list of (u, v) in cycles/pixel;
    the symmetric partner of each notch is removed automatically."""
    def build(s):
        u, v = _freq_axes(s)
        mask = np.ones(s)
        for cu, cv in centers:
            for sign in (1, -1):
                d2 = (u - sign * cu) ** 2 + (v - sign * cv) ** 2
                mask *= 1.0 - np.exp(-d2 / (2 * radius ** 2))
        return mask
    return build


def directional_blur(angle_deg, strength=0.05):
    """Blur along one direction (approximates motion blur)."""
    def build(s):
        u, v = _freq_axes(s)
        t = np.deg2rad(angle_deg)
        along = u * np.cos(t) + v * np.sin(t)
        return np.exp(-(along ** 2) / (2 * strength ** 2))
    return build


# ==========================================================================
# 3. Pillow <-> NumPy bridge
# ==========================================================================
def split_channels(img):
    """Return (list of float64 2D arrays, mode, alpha).
    Palette/CMYK/etc. are converted to RGB; alpha is kept aside untouched."""
    alpha = None
    if img.mode in ("RGBA", "LA"):
        alpha = img.getchannel("A")
        img = img.convert("RGB" if img.mode == "RGBA" else "L")
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:
        return [arr], img.mode, alpha
    return [arr[..., c] for c in range(arr.shape[2])], img.mode, alpha


def merge_channels(channels, mode, alpha=None):
    """Clip to 0..255 and rebuild a PIL image (re-attaching alpha if any)."""
    chans = [Image.fromarray(np.clip(np.rint(c), 0, 255).astype(np.uint8), "L")
             for c in channels]
    img = chans[0] if mode == "L" else Image.merge(mode, chans)
    if alpha is not None:
        img.putalpha(alpha)
    return img


def apply_to_image(img, mask_fn, pad=True):
    """Apply a frequency-domain mask to every color channel of a PIL image."""
    chans, mode, alpha = split_channels(img)
    return merge_channels([filter_channel(c, mask_fn, pad) for c in chans],
                          mode, alpha)


# ==========================================================================
# 4. High-level edits on PIL images
# ==========================================================================
def blur(img, cutoff=0.08):
    return apply_to_image(img, gaussian_lowpass(cutoff))


def sharpen(img, cutoff=0.1, amount=1.5):
    """Unsharp masking in frequency space:  H = 1 + amount * HighPass."""
    hp = gaussian_highpass(cutoff)
    return apply_to_image(img, lambda s: 1.0 + amount * hp(s))


def edge_detect(img, cutoff=0.1):
    """High-pass the luminance; result is a grayscale edge map."""
    gray = img.convert("L")
    arr = filter_channel(np.asarray(gray, np.float64), gaussian_highpass(cutoff))
    return merge_channels([arr + 128], "L")  # +128 so edges show as +/- gray


def bandpass_filter(img, low_cut=0.02, high_cut=0.2):
    return apply_to_image(img, bandpass(low_cut, high_cut))


def remove_periodic_noise(img, centers, radius=0.01):
    return apply_to_image(img, notch_reject(centers, radius))


def motion_blur(img, angle_deg=0, strength=0.05):
    return apply_to_image(img, directional_blur(angle_deg, strength))


def homomorphic(img, cutoff=0.05, gamma_low=0.5, gamma_high=1.8):
    """Even out uneven lighting and boost contrast. Works on the luminance
    (Y of YCbCr) of log(image) so colors are preserved."""
    ycc = img.convert("YCbCr")
    y, cb, cr = ycc.split()
    log_y = np.log1p(np.asarray(y, np.float64))
    hp = gaussian_highpass(cutoff)
    mask = lambda s: gamma_low + (gamma_high - gamma_low) * hp(s)
    new_y = np.expm1(filter_channel(log_y, mask))
    new_y = np.clip(np.rint(new_y), 0, 255).astype(np.uint8)
    out = Image.merge("YCbCr", (Image.fromarray(new_y, "L"), cb, cr))
    return out.convert("RGB")


def wiener_deblur(img, kernel, noise_to_signal=0.01):
    """Undo a known blur kernel (2D array summing to 1).
    H = conj(K) / (|K|^2 + NSR)."""
    chans, mode, alpha = split_channels(img)
    out = []
    for c in chans:
        K = np.fft.fft2(kernel, s=c.shape)
        W = np.conj(K) / (np.abs(K) ** 2 + noise_to_signal)
        out.append(np.real(np.fft.ifft2(np.fft.fft2(c) * W)))
    return merge_channels(out, mode, alpha)


def magnitude_spectrum(img):
    """Log-scaled spectrum as a grayscale PIL image (to choose mask settings)."""
    gray = np.asarray(img.convert("L"), np.float64)
    mag = np.log1p(np.abs(fft2c(gray)))
    return merge_channels([255 * mag / mag.max()], "L")


def phase_correlation_shift(img_a, img_b):
    """Estimate the (dy, dx) shift of img_a relative to img_b."""
    a = np.asarray(img_a.convert("L"), np.float64)
    b = np.asarray(img_b.convert("L").resize(img_a.size), np.float64)
    cross = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    cross /= np.abs(cross) + 1e-12
    corr = np.real(np.fft.ifft2(cross))
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return int(dy), int(dx)


# ==========================================================================
# 4b. More edits and effects
# ==========================================================================
def compress(img, keep=0.05):
    """Frequency-domain compression: keep only the strongest `keep` fraction
    of Fourier coefficients (the DC term is always among them)."""
    chans, mode, alpha = split_channels(img)
    out = []
    for c in chans:
        F = np.fft.fft2(c)
        mag = np.abs(F)
        k = max(1, int(keep * mag.size))
        thr = np.partition(mag.ravel(), -k)[-k]
        F[mag < thr] = 0
        out.append(np.real(np.fft.ifft2(F)))
    return merge_channels(out, mode, alpha)


def find_periodic_peaks(gray, n=4, min_sigma=6.0, ignore=0.02):
    """Locate the strongest isolated spikes in the spectrum of a 2D array.
    Returns up to n (u, v) positions in cycles/pixel. A spike counts only if
    it stands out from its local background by `min_sigma` robust sigmas."""
    rows, cols = gray.shape
    L = np.log1p(np.abs(fft2c(gray)))
    fy = np.fft.fftfreq(rows)[:, None]
    fx = np.fft.fftfreq(cols)[None, :]
    blur_k = np.exp(-2 * np.pi ** 2 * 8.0 ** 2 * (fx ** 2 + fy ** 2))
    diff = L - np.real(np.fft.ifft2(np.fft.fft2(L) * blur_k))
    cy, cx = rows // 2, cols // 2
    yy, xx = np.mgrid[0:rows, 0:cols]
    diff[np.hypot((yy - cy) / rows, (xx - cx) / cols) < ignore] = 0
    med = np.median(diff)
    z = (diff - med) / (1.4826 * np.median(np.abs(diff - med)) + 1e-12)
    peaks = []
    for _ in range(n):
        r, c = np.unravel_index(np.argmax(z), z.shape)
        if z[r, c] < min_sigma:
            break
        peaks.append(((c - cx) / cols, (r - cy) / rows))
        for rr, cc in ((r, c), ((2 * cy - r) % rows, (2 * cx - c) % cols)):
            z[max(rr - 3, 0):rr + 4, max(cc - 3, 0):cc + 4] = -np.inf
    return peaks


def auto_notch(img, peaks=4, radius=0.008, min_sigma=6.0):
    """Detect periodic noise (stripes, moire, screen patterns) and remove it."""
    gray = np.asarray(img.convert("L"), np.float64)
    centers = find_periodic_peaks(gray, peaks, min_sigma)
    return remove_periodic_noise(img, centers, radius) if centers else img.copy()


def equalizer(img, low=1.0, mid=1.0, high=1.0):
    """Frequency 'EQ': scale broad detail (low), medium detail (mid) and
    fine detail/noise (high). 1.0 = unchanged."""
    nodes, gains = [0.0, 0.03, 0.10, 0.25, 0.75], [low, low, mid, high, high]
    return apply_to_image(img, lambda s: np.interp(_freq_grid(s), nodes, gains))


def emboss(img, angle_deg=45.0, strength=2.0):
    """Directional derivative (multiply the spectrum by i*2*pi*f) -> relief look."""
    t = np.deg2rad(angle_deg)

    def mask(s):
        u, v = _freq_axes(s)
        return 1j * 2 * np.pi * (u * np.cos(t) + v * np.sin(t)) * strength

    gray = np.asarray(img.convert("L"), np.float64)
    return merge_channels([filter_channel(gray, mask) + 128], "L")


def deblur_gaussian(img, sigma=2.0, nsr=0.01):
    """Wiener deconvolution assuming a Gaussian blur of `sigma` pixels
    (nsr = noise-to-signal ratio; raise it for noisy images)."""
    def mask(s):
        K = np.exp(-2 * np.pi ** 2 * sigma ** 2 * _freq_grid(s) ** 2)
        return K / (K ** 2 + nsr)
    return apply_to_image(img, mask)


def hybrid(img_low, img_high, cutoff=0.05):
    """Hybrid image: coarse structure of one picture + fine detail of another.
    Looks like the first up close... and the other from far away (or vice versa)."""
    a, b = img_low.convert("RGB"), img_high.convert("RGB").resize(img_low.size)
    ca, cb = split_channels(a)[0], split_channels(b)[0]
    out = [filter_channel(x, gaussian_lowpass(cutoff)) +
           filter_channel(y, gaussian_highpass(cutoff)) for x, y in zip(ca, cb)]
    return merge_channels(out, "RGB")


def custom_mask(img, mask_img, invert=False):
    """Hand-drawn spectrum edit. Save magnitude_spectrum(img), paint on it
    (white = keep, black = remove; keep the center white and the drawing
    symmetric), then pass the painted image here."""
    def mask(s):
        m = np.asarray(mask_img.convert("L").resize((s[1], s[0])), np.float64) / 255.0
        return 1.0 - m if invert else m
    return apply_to_image(img, mask, pad=False)


# ==========================================================================
# 5. Registry of edits (single source of truth for CLI, batch mode and GUI)
#    name -> (function, {parameter: (default, min, max)})
#    An int default makes that parameter an integer everywhere.
# ==========================================================================
EDITS = {
    "blur":        (blur,              {"cutoff": (0.08, 0.005, 0.5)}),
    "sharpen":     (sharpen,           {"cutoff": (0.1, 0.01, 0.5), "amount": (1.5, 0.0, 5.0)}),
    "edge":        (edge_detect,       {"cutoff": (0.1, 0.01, 0.5)}),
    "bandpass":    (bandpass_filter,   {"low_cut": (0.02, 0.002, 0.5), "high_cut": (0.2, 0.01, 0.7)}),
    "motion":      (motion_blur,       {"angle_deg": (0.0, 0.0, 180.0), "strength": (0.05, 0.005, 0.5)}),
    "homomorphic": (homomorphic,       {"cutoff": (0.05, 0.005, 0.5), "gamma_low": (0.5, 0.1, 1.0),
                                        "gamma_high": (1.8, 1.0, 4.0)}),
    "deblur":      (deblur_gaussian,   {"sigma": (2.0, 0.5, 8.0), "nsr": (0.01, 0.0001, 0.2)}),
    "equalizer":   (equalizer,         {"low": (1.0, 0.0, 3.0), "mid": (1.0, 0.0, 3.0),
                                        "high": (1.0, 0.0, 3.0)}),
    "emboss":      (emboss,            {"angle_deg": (45.0, 0.0, 360.0), "strength": (2.0, 0.1, 10.0)}),
    "compress":    (compress,          {"keep": (0.05, 0.001, 1.0)}),
    "auto_notch":  (auto_notch,        {"peaks": (4, 1, 16), "radius": (0.008, 0.002, 0.05),
                                        "min_sigma": (6.0, 2.0, 20.0)}),
    "spectrum":    (magnitude_spectrum, {}),
}
_ALIASES = {"edge_detect": "edge", "motion_blur": "motion"}


def parse_step(spec):
    """'sharpen:amount=2,cutoff=0.2' -> ('sharpen', {'amount': 2.0, 'cutoff': 0.2})"""
    name, _, argstr = spec.partition(":")
    name = name.strip()
    if name not in EDITS:
        raise ValueError(f"unknown edit '{name}' (available: {', '.join(EDITS)})")
    params = EDITS[name][1]
    kwargs = {}
    for part in filter(None, (p.strip() for p in argstr.split(","))):
        key, eq, val = part.partition("=")
        key = key.strip()
        if not eq or key not in params:
            raise ValueError(f"bad setting '{part}' for '{name}' "
                             f"(settings: {', '.join(params) or 'none'})")
        kwargs[key] = int(float(val)) if isinstance(params[key][0], int) else float(val)
    return name, kwargs


def run_pipeline(img, steps):
    """Apply [(name, kwargs), ...] to a PIL image, in order."""
    for name, kwargs in steps:
        img = EDITS[name][0](img, **kwargs)
    return img


# ==========================================================================
# 6. Chainable editor class
# ==========================================================================
class FFTEditor:
    """Wraps a PIL image; each edit returns a new FFTEditor so calls chain.
    Every name in EDITS is available as a method, e.g. .emboss(angle_deg=30)."""

    def __init__(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError("FFTEditor expects a PIL.Image.Image")
        self.image = image

    @classmethod
    def open(cls, path):
        return cls(ImageOps.exif_transpose(Image.open(path)))

    def save(self, path, **kwargs):
        save_image(self.image, path, **kwargs)
        return self

    def apply(self, name, **kwargs):
        return FFTEditor(EDITS[_ALIASES.get(name, name)][0](self.image, **kwargs))

    def pipeline(self, steps):
        return FFTEditor(run_pipeline(self.image, steps))

    def __getattr__(self, name):
        if name in EDITS or name in _ALIASES:
            return lambda **kw: self.apply(name, **kw)
        raise AttributeError(name)

    # edits that need extra arguments
    def remove_periodic_noise(self, centers, **kw):
        return FFTEditor(remove_periodic_noise(self.image, centers, **kw))

    def wiener_deblur(self, kernel, **kw):
        return FFTEditor(wiener_deblur(self.image, kernel, **kw))

    def hybrid_with(self, other, **kw):
        return FFTEditor(hybrid(self.image, other, **kw))

    def custom_mask(self, mask_img, **kw):
        return FFTEditor(custom_mask(self.image, mask_img, **kw))


# ==========================================================================
# 7. Saving and batch processing
# ==========================================================================
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def save_image(img, path, **kwargs):
    """Save, creating folders as needed and converting modes JPEG can't hold."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        kwargs.setdefault("quality", 95)
    img.save(path, **kwargs)


def _process_one(task):
    src, dst, steps = task
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            im.load()
        save_image(run_pipeline(im, steps), dst)
        return str(src), None
    except Exception as exc:  # keep going; report at the end
        return str(src), f"{type(exc).__name__}: {exc}"


def batch_process(in_dir, out_dir, steps, recursive=False, suffix="", fmt=None,
                  jobs=1, skip_existing=False, progress=None):
    """Run a pipeline over every image in a folder (folder structure is kept).
    Returns [(source, error_or_None), ...]. progress(i, total, result) is
    called after each file."""
    in_dir, out_dir = Path(in_dir).resolve(), Path(out_dir).resolve()
    candidates = in_dir.rglob("*") if recursive else in_dir.glob("*")
    tasks = []
    for src in sorted(candidates):
        if not (src.is_file() and src.suffix.lower() in IMAGE_EXTS):
            continue
        if out_dir in src.resolve().parents:      # never reprocess our own output
            continue
        rel = src.relative_to(in_dir)
        ext = "." + fmt.lstrip(".") if fmt else src.suffix
        dst = out_dir / rel.parent / (rel.stem + suffix + ext)
        if skip_existing and dst.exists():
            continue
        tasks.append((src, dst, steps))

    results = []

    def record(result):
        results.append(result)
        if progress:
            progress(len(results), len(tasks), result)

    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(jobs) as pool:
            for r in pool.map(_process_one, tasks):
                record(r)
    else:
        for t in tasks:
            record(_process_one(t))
    return results


# ==========================================================================
# 8. Command-line interface
# ==========================================================================
def _list_edits():
    for name, (func, params) in EDITS.items():
        doc = (func.__doc__ or "").strip().splitlines()
        print(f"{name:12s} {doc[0] if doc else ''}")
        for key, (default, lo, hi) in params.items():
            print(f"{'':14s}{key} = {default}   (range {lo} .. {hi})")


def _cli(argv=None):
    p = argparse.ArgumentParser(
        description="FFT-based image editing (Pillow + NumPy). Single file or whole folder.",
        epilog="examples:\n"
               "  %(prog)s in.jpg out.png -e sharpen:amount=2 -e blur:cutoff=0.3\n"
               "  %(prog)s ./photos ./edited -e auto_notch -r -j 4 --format png",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", nargs="?", help="image file or folder")
    p.add_argument("output", nargs="?", help="output file (or folder, for a folder input)")
    p.add_argument("-e", "--edit", action="append", default=[], metavar="NAME[:k=v,...]",
                   help="edit to apply; repeat to chain (see --list)")
    p.add_argument("-r", "--recursive", action="store_true", help="include subfolders")
    p.add_argument("-j", "--jobs", type=int, default=1, help="parallel worker processes")
    p.add_argument("--suffix", default="", help="text added to output file names")
    p.add_argument("--format", help="output format for folder mode, e.g. png")
    p.add_argument("--skip-existing", action="store_true", help="don't redo finished files")
    p.add_argument("--list", action="store_true", help="list edits and settings")
    p.add_argument("--gui", action="store_true", help="open the interactive editor")
    p.add_argument("--selftest", action="store_true", help="run the built-in tests")
    a = p.parse_args(argv)

    if a.selftest:
        return _selftest()
    if a.list:
        return _list_edits()
    if a.gui:
        import fft_image_gui
        return fft_image_gui.main()
    if not (a.input and a.output and a.edit):
        p.error("need INPUT, OUTPUT and at least one -e EDIT (or use --list / --gui)")
    try:
        steps = [parse_step(s) for s in a.edit]
    except ValueError as exc:
        p.error(str(exc))

    if Path(a.input).is_dir():
        def show(i, n, r):
            print(f"[{i}/{n}] {'OK  ' if r[1] is None else 'FAIL'} {r[0]}"
                  + (f"  -> {r[1]}" if r[1] else ""))
        results = batch_process(a.input, a.output, steps, a.recursive, a.suffix,
                                a.format, a.jobs, a.skip_existing, show)
        failed = sum(1 for _, e in results if e)
        print(f"Done: {len(results) - failed} processed, {failed} failed.")
        sys.exit(1 if failed else 0)
    src, err = _process_one((a.input, a.output, steps))
    if err:
        sys.exit(f"error: {err}")
    print(f"Saved {a.output}")


# ==========================================================================
# 9. Self-test (python fft_image_editor.py --selftest)
# ==========================================================================
def _selftest():
    import contextlib
    import io
    import tempfile
    rng = np.random.default_rng(0)
    N = 256
    y, x = np.mgrid[0:N, 0:N]
    base = 128 + 80 * np.sin(x / 15.0) * np.cos(y / 20.0)
    base[100:160, 100:160] += 60
    gray = merge_channels([base], "L")
    rgb = merge_channels([base, base * 0.8, base * 0.6], "RGB")
    rgba = rgb.copy()
    rgba.putalpha(Image.new("L", rgb.size, 200))

    energy = lambda im: (np.diff(np.asarray(im.convert("L"), np.float64), axis=1) ** 2).mean()
    err = lambda a, b: np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).mean()

    # --- original edits -----------------------------------------------------
    for im in (gray, rgb, rgba):
        out = blur(im)
        assert out.mode == im.mode and out.size == im.size, "mode/size changed"
    assert np.all(np.asarray(blur(rgba).getchannel("A")) == 200), "alpha lost"
    assert err(apply_to_image(rgb, lambda s: np.ones(s)), rgb) < 0.5, "identity filter"
    assert energy(blur(rgb)) < energy(rgb) < energy(sharpen(rgb)), "blur/sharpen"

    ref = merge_channels([base] * 3, "RGB")
    noisy = merge_channels([base + 25 * np.sin(2 * np.pi * 0.203125 * x)] * 3, "RGB")
    clean = remove_periodic_noise(noisy, [(0.203125, 0.0)], radius=0.01)
    assert err(clean, ref) < err(noisy, ref) / 3, "notch filter"

    shifted = Image.fromarray(np.roll(np.asarray(gray), (7, -12), axis=(0, 1)))
    assert phase_correlation_shift(shifted, gray) == (7, -12), "phase corr"

    k = np.ones((1, 9)) / 9.0
    b = np.real(np.fft.ifft2(np.fft.fft2(base) * np.fft.fft2(k, s=base.shape)))
    blurred = merge_channels([b + rng.normal(0, 0.5, base.shape)], "L")
    assert err(wiener_deblur(blurred, k, 0.001), gray) < err(blurred, gray), "wiener"

    # --- new edits ----------------------------------------------------------
    assert err(compress(rgb, keep=1.0), rgb) < 0.5, "compress keep=1 should be lossless"
    assert err(compress(rgb, 0.02), rgb) > err(compress(rgb, 0.3), rgb), "compress ordering"

    peaks = find_periodic_peaks(np.asarray(noisy.convert("L"), np.float64), n=2)
    assert any(abs(abs(u) - 0.203125) < 0.01 and abs(v) < 0.01 for u, v in peaks), \
        f"peak detection found {peaks}"
    assert err(auto_notch(noisy, peaks=2), ref) < err(noisy, ref) / 2, "auto_notch"
    assert auto_notch(rgb, peaks=2, min_sigma=50).size == rgb.size, "auto_notch no-peak path"

    assert err(equalizer(rgb), rgb) < 0.5, "equalizer identity"
    assert energy(equalizer(rgb, high=2.5)) > energy(rgb), "equalizer boost"

    emb = emboss(rgb)
    assert emb.mode == "L" and abs(np.asarray(emb).mean() - 128) < 5, "emboss"

    fine = merge_channels([128 + 60 * np.sin(x / 1.5) * np.sin(y / 1.5) + 40 * (x > 128)] * 3, "RGB")
    soft = blur(fine, 0.08)                     # Gaussian blur of sigma ~ 1/(2*pi*0.08) px
    assert err(deblur_gaussian(soft, 1.99, 0.001), fine) < err(soft, fine) * 0.7, "deblur_gaussian"

    hy = hybrid(rgb, merge_channels([255 - base] * 3, "RGB"))
    assert hy.mode == "RGB" and hy.size == rgb.size and err(hy, rgb) > 1, "hybrid"

    white, black = Image.new("L", rgb.size, 255), Image.new("L", rgb.size, 0)
    assert err(custom_mask(rgb, white), rgb) < 0.5, "custom_mask keep-all"
    assert np.asarray(custom_mask(rgb, black)).max() == 0, "custom_mask remove-all"

    # --- registry / pipeline ------------------------------------------------
    assert parse_step("sharpen:amount=2,cutoff=0.2") == ("sharpen", {"amount": 2.0, "cutoff": 0.2})
    assert parse_step("auto_notch:peaks=3") == ("auto_notch", {"peaks": 3})
    for bad in ("nope", "blur:wrong=1", "blur:cutoff"):
        try:
            parse_step(bad)
            raise AssertionError(f"parse_step accepted {bad!r}")
        except ValueError:
            pass
    steps = [parse_step("sharpen"), parse_step("blur:cutoff=0.3")]
    assert err(run_pipeline(rgb, steps), FFTEditor(rgb).sharpen().blur(cutoff=0.3).image) == 0
    assert FFTEditor(rgb).edge_detect().image.mode == "L", "alias method"
    for name, (func, params) in EDITS.items():  # every registered edit must run on defaults
        out = func(rgb, **{k: v[0] for k, v in params.items()})
        assert out.size == rgb.size, name

    # --- batch mode ---------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in" / "sub").mkdir(parents=True)
        rgb.save(tmp / "in" / "a.png")
        gray.save(tmp / "in" / "sub" / "b.jpg")
        rgba.save(tmp / "in" / "sub" / "c.png")
        (tmp / "in" / "notes.txt").write_text("not an image")
        res = batch_process(tmp / "in", tmp / "out", steps, recursive=True, suffix="_fft", jobs=1)
        assert len(res) == 3 and all(e is None for _, e in res), res
        assert (tmp / "out" / "a_fft.png").exists() and (tmp / "out" / "sub" / "b_fft.jpg").exists()
        assert len(batch_process(tmp / "in", tmp / "out", steps, False, "_fft")) == 1, "non-recursive"
        assert not batch_process(tmp / "in", tmp / "out", steps, True, "_fft", skip_existing=True)
        res = batch_process(tmp / "in", tmp / "out2", steps, True, fmt="webp", jobs=2)
        assert len(res) == 3 and all(e is None for _, e in res), "parallel batch"
        # output folder nested in the input folder must not be reprocessed
        res = batch_process(tmp / "in", tmp / "in" / "edited", steps, recursive=True)
        res2 = batch_process(tmp / "in", tmp / "in" / "edited", steps, recursive=True)
        assert len(res) == len(res2) == 3, "reprocessed own output"
        # CLI single file
        with contextlib.redirect_stdout(io.StringIO()):
            _cli([str(tmp / "in" / "a.png"), str(tmp / "cli.jpg"), "-e", "emboss:angle_deg=30"])
        assert (tmp / "cli.jpg").exists(), "cli single file"

    print("All FFT image-editing checks passed.")


if __name__ == "__main__":
    _cli()
