"""
fft_image_gui.py
Interactive front end for fft_image_editor (Tkinter + Pillow).

    python fft_image_gui.py          (or: python fft_image_editor.py --gui)

Build a pipeline of edits, tune each with sliders (live preview on a downscaled
copy), type exact values, save the full-resolution result, or run the same
pipeline over a folder. Controls are generated from fft_image_editor.EDITS, so
new edits appear here automatically.
"""
import math
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

import fft_image_editor as fe

PREVIEW_MAX = (900, 650)
OPEN_TYPES = [("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"), ("All files", "*.*")]


# ------------------------------------------------------------ slider mapping
# Sliders always run 0..1. Settings whose range spans a factor of 20 or more
# (cutoffs, noise ratios...) use a logarithmic scale so every part of the
# track is useful; other settings are linear.
def is_log(lo, hi):
    return lo > 0 and hi / lo >= 20


def slider_value(default, lo, hi, t):
    v = lo * (hi / lo) ** t if is_log(lo, hi) else lo + (hi - lo) * t
    return int(round(v)) if isinstance(default, int) else v


def slider_pos(lo, hi, v):
    t = math.log(v / lo) / math.log(hi / lo) if is_log(lo, hi) else (v - lo) / (hi - lo)
    return min(max(t, 0.0), 1.0)


def fmt(value):
    return str(value) if isinstance(value, int) else f"{value:.3g}"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FFT Image Editor")
        self.geometry("1280x780")
        self.original = None        # full-resolution image
        self.preview_src = None     # downscaled copy used for live preview
        self.photo = None           # keeps the Tk image alive
        self.steps = []             # [{"name": str, "params": {key: value}}]
        self._after = None          # pending (debounced) refresh
        self._want = None           # newest render request: (steps, show_original)
        self._busy = False          # a render thread is running
        self._result = None         # set by the render thread: (image, error)
        self.show_orig = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Open an image to begin.")
        self._build()

    # ---------------------------------------------------------------- layout
    def _build(self):
        left = ttk.Frame(self, padding=8)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=8)
        right.pack(side="left", fill="both", expand=True)

        row = ttk.Frame(left)
        row.pack(fill="x")
        for text, cmd in (("Open…", self.open_image), ("Save…", self.save_result),
                          ("Batch folder…", self.batch_dialog)):
            ttk.Button(row, text=text, command=cmd).pack(side="left", padx=2)

        ttk.Label(left, text="Add an edit:").pack(anchor="w", pady=(12, 0))
        row = ttk.Frame(left)
        row.pack(fill="x")
        self.combo = ttk.Combobox(row, values=list(fe.EDITS), state="readonly", width=16)
        self.combo.current(0)
        self.combo.pack(side="left")
        ttk.Button(row, text="Add", command=self.add_step).pack(side="left", padx=4)

        ttk.Label(left, text="Pipeline (top to bottom):").pack(anchor="w", pady=(12, 0))
        self.listbox = tk.Listbox(left, height=8, exportselection=False)
        self.listbox.pack(fill="x")
        self.listbox.bind("<<ListboxSelect>>", lambda e: self.show_params())
        row = ttk.Frame(left)
        row.pack(fill="x", pady=4)
        for text, cmd in (("Up", lambda: self.move(-1)), ("Down", lambda: self.move(1)),
                          ("Remove", self.remove_step)):
            ttk.Button(row, text=text, command=cmd, width=8).pack(side="left", padx=2)

        self.params = ttk.LabelFrame(left, text="Settings (drag, or type a value + Enter)",
                                     padding=8)
        self.params.pack(fill="x", pady=8)

        ttk.Checkbutton(right, text="Show original", variable=self.show_orig,
                        command=self.schedule).pack(anchor="w")
        self.view = ttk.Label(right, anchor="center")
        self.view.pack(fill="both", expand=True)
        ttk.Label(right, textvariable=self.status).pack(anchor="w")

    # ----------------------------------------------------------- image files
    def open_image(self):
        path = filedialog.askopenfilename(filetypes=OPEN_TYPES)
        if not path:
            return
        try:
            img = ImageOps.exif_transpose(Image.open(path))
            img.load()
        except Exception as exc:
            messagebox.showerror("Could not open image", str(exc))
            return
        self.original = img
        self.preview_src = img.copy()
        self.preview_src.thumbnail(PREVIEW_MAX)
        self.status.set(f"{path}  ({img.width}x{img.height})")
        self.schedule(delay=0)

    def save_result(self):
        if self.original is None:
            return messagebox.showinfo("Nothing to save", "Open an image first.")
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("WebP", "*.webp"), ("TIFF", "*.tif")])
        if not path:
            return
        self.status.set("Processing full-resolution image…")
        self.update_idletasks()
        try:
            fe.save_image(fe.run_pipeline(self.original, self.steps_as_tuples()), path)
            self.status.set(f"Saved {path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    # -------------------------------------------------------------- pipeline
    def steps_as_tuples(self):
        return [(s["name"], dict(s["params"])) for s in self.steps]

    def selected(self):
        sel = self.listbox.curselection()
        return sel[0] if sel else None

    def reload_list(self, select=None):
        self.listbox.delete(0, "end")
        for i, s in enumerate(self.steps, 1):
            self.listbox.insert("end", f"{i}. {s['name']}")
        if select is not None:
            self.listbox.selection_set(select)

    def add_step(self):
        name = self.combo.get()
        defaults = {k: v[0] for k, v in fe.EDITS[name][1].items()}
        self.steps.append({"name": name, "params": defaults})
        self.reload_list(len(self.steps) - 1)
        self.show_params()
        self.schedule(delay=0)

    def remove_step(self):
        idx = self.selected()
        if idx is None:
            return
        self.steps.pop(idx)
        self.reload_list(min(idx, len(self.steps) - 1) if self.steps else None)
        self.show_params()
        self.schedule(delay=0)

    def move(self, delta):
        idx = self.selected()
        new = None if idx is None else idx + delta
        if new is None or not 0 <= new < len(self.steps):
            return
        self.steps[idx], self.steps[new] = self.steps[new], self.steps[idx]
        self.reload_list(new)
        self.schedule(delay=0)

    # -------------------------------------------------------------- settings
    def show_params(self):
        for child in self.params.winfo_children():
            child.destroy()
        idx = self.selected()
        if idx is None:
            return
        step = self.steps[idx]
        spec = fe.EDITS[step["name"]][1]
        if not spec:
            ttk.Label(self.params, text="This edit has no settings.").pack()
        for key, (default, lo, hi) in spec.items():
            self._add_slider(step, key, default, lo, hi)

    def _add_slider(self, step, key, default, lo, hi):
        row = ttk.Frame(self.params)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=key, width=11).pack(side="left")
        text = tk.StringVar(value=fmt(step["params"][key]))
        entry = ttk.Entry(row, textvariable=text, width=8)
        entry.pack(side="right")

        def on_slide(raw):
            value = slider_value(default, lo, hi, float(raw))
            step["params"][key] = value
            text.set(fmt(value))
            self.schedule()

        def on_type(event=None):
            try:
                value = min(max(float(text.get()), lo), hi)
            except ValueError:
                text.set(fmt(step["params"][key]))
                return
            value = int(round(value)) if isinstance(default, int) else value
            step["params"][key] = value
            scale.set(slider_pos(lo, hi, value))   # moves the handle (fires on_slide)
            step["params"][key] = value            # keep the exact typed value
            text.set(fmt(value))
            self.schedule(delay=0)

        scale = ttk.Scale(row, from_=0.0, to=1.0, length=150,
                          value=slider_pos(lo, hi, step["params"][key]), command=on_slide)
        scale.pack(side="left", fill="x", expand=True)
        scale.bind("<ButtonRelease-1>", lambda e: self.schedule(delay=0))
        entry.bind("<Return>", on_type)
        entry.bind("<FocusOut>", on_type)

    # --------------------------------------------------------------- preview
    def schedule(self, delay=40):
        """Request a re-render. Rapid requests (slider drags) are merged."""
        if self._after:
            self.after_cancel(self._after)
        self._after = self.after(delay, self.refresh)

    def refresh(self):
        """Queue a render. It runs in a worker thread so dragging a slider never
        freezes the window; only the newest request is rendered."""
        self._after = None
        if self.preview_src is None:
            return
        self._want = (self.steps_as_tuples(), self.show_orig.get())
        if not self._busy:
            self._start_render()

    def _start_render(self):
        (steps, show_orig), self._want = self._want, None
        self._busy, self._result = True, None
        src = self.preview_src

        def work():
            try:
                self._result = (src if show_orig else fe.run_pipeline(src, steps), None)
            except Exception as exc:
                self._result = (None, str(exc))

        threading.Thread(target=work, daemon=True).start()
        self.after(20, self._poll_render)

    def _poll_render(self):
        if self._result is None:
            return self.after(20, self._poll_render)
        img, err = self._result
        self._busy = False
        if self._want is not None:          # a newer request arrived meanwhile
            self._start_render()
            if err is None:
                self._show(img)             # still show this frame for feedback
            return
        if err:
            self.status.set(f"Error: {err}")
        else:
            self._show(img)

    def _show(self, img):
        self.photo = ImageTk.PhotoImage(img)
        self.view.config(image=self.photo)

    # ----------------------------------------------------------------- batch
    def batch_dialog(self):
        if not self.steps:
            return messagebox.showinfo("Empty pipeline", "Add at least one edit first.")
        in_dir = filedialog.askdirectory(title="Folder with images")
        out_dir = in_dir and filedialog.askdirectory(title="Output folder")
        if not in_dir or not out_dir:
            return
        recursive = messagebox.askyesno("Subfolders", "Include images in subfolders?")
        steps = self.steps_as_tuples()
        state = {"done": 0, "total": 0, "errors": [], "finished": False}

        def progress(i, n, result):
            state.update(done=i, total=n)
            if result[1]:
                state["errors"].append(result)

        def work():
            try:
                fe.batch_process(in_dir, out_dir, steps, recursive=recursive, progress=progress)
            except Exception as exc:
                state["errors"].append(("batch", str(exc)))
            state["finished"] = True

        threading.Thread(target=work, daemon=True).start()
        self.poll_batch(state)

    def poll_batch(self, state):
        self.status.set(f"Batch: {state['done']}/{state['total'] or '?'} images")
        if not state["finished"]:
            return self.after(200, lambda: self.poll_batch(state))
        msg = f"Processed {state['done']} image(s), {len(state['errors'])} error(s)."
        if state["errors"]:
            msg += "\n\n" + "\n".join(f"{s}: {e}" for s, e in state["errors"][:5])
        self.status.set(msg.splitlines()[0])
        messagebox.showinfo("Batch finished", msg)


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
