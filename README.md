# FFT Image Editor

Edit images in the **frequency domain** using the Fast Fourier Transform. Blur, sharpen, remove stripe noise, fix uneven lighting, deblur, emboss, and more, all from one small, dependency-light Python project built on NumPy and Pillow.

Use it three ways: as a **Python library**, as a **command-line tool** (single files or whole folders), or through an **interactive GUI** with live preview.

## How it works

Filtering an image in the spatial domain is the same as multiplying its spectrum by a mask (the convolution theorem). Every edit in this project follows the same pipeline, and only the mask changes:

```
PIL image -> array -> FFT -> multiply by mask H(u,v) -> inverse FFT -> PIL image
```

A low-pass mask blurs, a high-pass mask sharpens or finds edges, a notch mask deletes a single spike (periodic noise), and so on. Adding a new effect means writing one mask function.

## Files

| File | Purpose |
| --- | --- |
| `fft_image_editor.py` | The library, command-line tool, batch processor, and self-tests |
| `fft_image_gui.py` | Tkinter GUI with live preview (uses the library) |
| `requirements.txt` | Third-party packages to install with pip |

## Installation

```bash
pip install -r requirements.txt     # or: pip install numpy pillow
```

Developed and tested with Python 3.12. The code uses no recent language features, so older Python 3 versions should work as long as a compatible NumPy and Pillow are available.

No other setup is needed. Copy the files into your project, or run them where they are.

## Dependencies

### Third-party packages

| Package | Used for | Required by |
| --- | --- | --- |
| [NumPy](https://numpy.org) | The FFT itself (`numpy.fft`), building the frequency masks, and all array math | `fft_image_editor.py` |
| [Pillow](https://python-pillow.org) (`PIL`) | Opening, converting and saving images (`Image`), EXIF orientation (`ImageOps`), and showing the preview in the GUI (`ImageTk`) | both files |

Tested with NumPy 2.5.4 and Pillow 13.0.0.

### System package (GUI only)

| Package | Used for | Install |
| --- | --- | --- |
| `tkinter` | The window, sliders, dialogs and file pickers in `fft_image_gui.py` | Included with most Python installers. On Debian/Ubuntu: `sudo apt install python3-tk` |

The command-line tool and the Python library do not need `tkinter`.

### Python standard library

No extra install needed. These built-in modules are used:

| Module | Used for |
| --- | --- |
| `argparse` | Command-line options |
| `pathlib` | File and folder paths in batch mode |
| `concurrent.futures` | Parallel batch processing (`ProcessPoolExecutor`, the `-j` option) |
| `threading` | Background preview rendering and batch runs in the GUI, so the window never freezes |
| `math` | Logarithmic slider mapping in the GUI |
| `sys` | Exit codes and error output |
| `tempfile`, `contextlib`, `io` | The built-in self-test only |

## Quick start

```bash
# one image, two chained edits
python fft_image_editor.py photo.jpg out.png -e sharpen:amount=2 -e blur:cutoff=0.3

# a whole folder, including subfolders, 4 worker processes
python fft_image_editor.py ./photos ./edited -e auto_notch -e sharpen -r -j 4 --format png

# open the GUI
python fft_image_editor.py --gui
```

## Edits

List them any time with `python fft_image_editor.py --list`.

| Edit | What it does | Settings (default, range) |
| --- | --- | --- |
| `blur` | Gaussian low-pass blur | `cutoff` (0.08, 0.005 to 0.5) |
| `sharpen` | Unsharp masking in frequency space | `cutoff` (0.1, 0.01 to 0.5), `amount` (1.5, 0 to 5) |
| `edge` | High-pass edge map (grayscale) | `cutoff` (0.1, 0.01 to 0.5) |
| `bandpass` | Keep only a band of detail sizes | `low_cut` (0.02, 0.002 to 0.5), `high_cut` (0.2, 0.01 to 0.7) |
| `motion` | Directional (motion-style) blur | `angle_deg` (0, 0 to 180), `strength` (0.05, 0.005 to 0.5) |
| `homomorphic` | Even out lighting and boost contrast (works on luminance, keeps colors) | `cutoff` (0.05, 0.005 to 0.5), `gamma_low` (0.5, 0.1 to 1), `gamma_high` (1.8, 1 to 4) |
| `deblur` | Wiener deconvolution assuming a Gaussian blur | `sigma` (2, 0.5 to 8), `nsr` (0.01, 0.0001 to 0.2) |
| `equalizer` | Frequency "EQ": scale coarse, medium and fine detail | `low`, `mid`, `high` (each 1, 0 to 3) |
| `emboss` | Directional derivative, relief look (grayscale) | `angle_deg` (45, 0 to 360), `strength` (2, 0.1 to 10) |
| `compress` | Keep only the strongest fraction of Fourier coefficients | `keep` (0.05, 0.001 to 1) |
| `auto_notch` | Detect and remove periodic noise (stripes, moire) automatically | `peaks` (4, 1 to 16), `radius` (0.008, 0.002 to 0.05), `min_sigma` (6, 2 to 20) |
| `spectrum` | Log-scaled spectrum image, useful for choosing mask settings | none |

**Units:** frequency settings (`cutoff`, `low_cut`, `high_cut`, `strength` for `motion`, `radius`) are in **cycles per pixel**. The maximum is 0.5, and smaller cutoffs mean stronger blur. `sigma` is in pixels.

### Extra functions (Python only)

These need more than numbers, so they are not in the CLI or GUI:

- `remove_periodic_noise(img, centers, radius)`: notch out specific spectrum positions you choose.
- `wiener_deblur(img, kernel, noise_to_signal)`: undo a known blur kernel.
- `hybrid(img_low, img_high, cutoff)`: coarse structure of one image plus fine detail of another.
- `custom_mask(img, mask_img, invert)`: hand-drawn spectrum edit. Save `magnitude_spectrum(img)`, paint on it (white keeps, black removes; keep the center white and the drawing symmetric), then pass the painted image in.
- `phase_correlation_shift(img_a, img_b)`: estimate the `(dy, dx)` translation between two images.

## Command line

```
python fft_image_editor.py [input] [output] [options]
```

| Option | Meaning |
| --- | --- |
| `-e NAME[:k=v,...]` | Edit to apply. Repeat to chain edits in order, e.g. `-e sharpen:amount=2,cutoff=0.2 -e blur` |
| `-r`, `--recursive` | Include subfolders (folder mode) |
| `-j N`, `--jobs N` | Number of parallel worker processes |
| `--suffix TEXT` | Text added to output file names, e.g. `_edited` |
| `--format EXT` | Output format in folder mode, e.g. `png` or `webp` |
| `--skip-existing` | Skip files whose output already exists |
| `--list` | List all edits and their settings |
| `--gui` | Open the GUI |
| `--selftest` | Run the built-in tests |

If `input` is a folder, `output` is a folder and the subfolder structure is kept. Supported formats: PNG, JPEG, BMP, TIFF, WebP. In folder mode a failure on one file does not stop the run, and a summary is printed at the end (exit code 1 if anything failed). An output folder placed inside the input folder is never reprocessed.

## GUI

```bash
python fft_image_gui.py
```

- **Open / Save:** the live preview runs on a downscaled copy, and **Save** processes the full-resolution image.
- **Pipeline:** add edits, reorder them with Up and Down, remove them. They apply top to bottom.
- **Settings:** drag a slider or type an exact value and press Enter. Settings that span a wide range (like `cutoff`) use a logarithmic slider so the whole track is useful.
- **Show original:** toggle to compare before and after.
- **Batch folder:** run the current pipeline over a folder in the background.

Controls are generated from the registry (see below), so new edits appear in the GUI automatically.

## Python API

```python
from PIL import Image
from fft_image_editor import FFTEditor, blur, sharpen, run_pipeline, parse_step

# chainable editor: every registered edit is a method
out = (FFTEditor(Image.open("photo.jpg"))
       .sharpen(cutoff=0.1, amount=2)
       .blur(cutoff=0.3)
       .image)                      # a regular PIL.Image
out.save("result.png")

# plain functions on PIL images
soft = blur(Image.open("photo.jpg"), cutoff=0.05)

# a pipeline described as data
steps = [parse_step("auto_notch"), parse_step("sharpen:amount=2")]
result = run_pipeline(Image.open("photo.jpg"), steps)

# batch processing from code
from fft_image_editor import batch_process
batch_process("./photos", "./edited", steps, recursive=True, jobs=4)
```

Notes on image handling:

- RGB and grayscale images keep their mode and size. In RGBA images the alpha channel is set aside and re-attached untouched.
- Other modes (palette, CMYK, ...) are converted to RGB first.
- Each channel is **reflect-padded** before the FFT and cropped afterward, which avoids the dark or bright seams that the FFT's wrap-around assumption would otherwise cause at the borders.

## Adding your own edit

1. Write a function that takes a PIL image and returns a PIL image. Build it from a mask function and `apply_to_image(img, mask_fn)`; a mask function takes the array shape and returns an array of the same shape.
2. Add one line to the `EDITS` registry in `fft_image_editor.py`:

   ```python
   "myedit": (my_function, {"strength": (1.0, 0.0, 5.0)}),   # name: (default, min, max)
   ```

The edit is now available in the command line, batch mode, `FFTEditor`, and the GUI. An integer default (such as `4`) makes that setting an integer everywhere. A range spanning a factor of 20 or more gets a logarithmic slider.

## Testing

```bash
python fft_image_editor.py --selftest
```

The self-test checks mode and alpha handling, that an all-pass mask is lossless, that blur and sharpen change high-frequency energy in the right direction, notch and automatic periodic-noise removal, deblurring, phase correlation, every registered edit on its default settings, and batch mode (recursive, skip-existing, parallel workers, and a nested output folder).

## Limitations

- **Preview vs. saved result:** the GUI preview is a downscaled copy (up to 900x650). Since cutoffs are measured in cycles per pixel, a very large photo will look less blurred or sharpened in the saved result than in the preview for the same settings.
- **8-bit between steps:** images are converted to 8-bit after each edit, so chained edits round at every step, and very subtle effects can be lost.
- **Ringing:** sharp (ideal) filters cause visible ringing. The built-in edits use smooth Gaussian masks to avoid it.
- **Speed:** each edit costs one FFT per channel (O(N log N)). For very large images, swapping `numpy.fft` for `scipy.fft` or `pyfftw` with multithreading would be faster.
- **Known blur only:** `deblur` and `wiener_deblur` need the blur's shape, and they amplify noise if `nsr` is set too low.
