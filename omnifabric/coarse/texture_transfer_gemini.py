"""
Choose how the textured frontal view R (views/front_garment.png) is generated:

  [1] Gemini / Nano Banana (manual, as in the paper): this script prepares a hand-off folder with the grey garment render,
      the reference photo and the prompt; the user generates the image in the Gemini app and saves it into
      <handoff_dir>/result/, and the script imports it.
  [2] FLUX.2-dev (automatic): scripts/run_pipeline.sh then runs omnifabric.coarse.texture_transfer.

--mode ask      interactive menu (scripts/run_pipeline.sh uses it when run from a terminal)
--mode gemini   prepare the hand-off folder and import <handoff_dir>/result/ if an image is there; otherwise print the
                instructions and exit with code 3, so that re-running the pipeline later picks the image up
--mode flux2    do nothing (FLUX.2 runs afterwards)

Gemini keeps the garment's shape but often not its scale (e.g. it draws a small T-shirt larger), so the imported image
is chroma-keyed, its garment is scaled and shifted onto the grey render's silhouette (best IoU), and it is pasted onto
the render's green background, so the outline always matches the simulated garment (omnifabric/utils/fit_view.py).
Exit codes: 0 = done (output written, or FLUX.2 chosen: output not written), 3 = stopped, waiting for the Gemini image.
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
from PIL import Image

PROMPT = ("The grey apparel is my target apparel. Please generate textures on this grey apparel, so the textures are "
          "100% identical to the apparel in the reference image. Please don't change the style, size and pose of the "
          "grey garment.")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
WAITING = 3


def prepare_handoff(grey, reference, handoff_dir):
    os.makedirs(os.path.join(handoff_dir, "result"), exist_ok=True)
    shutil.copy(grey, os.path.join(handoff_dir, "1_grey_garment.png"))
    reference_copy = os.path.join(handoff_dir, "2_reference" + os.path.splitext(reference)[1].lower())
    shutil.copy(reference, reference_copy)
    with open(os.path.join(handoff_dir, "PROMPT.txt"), "w") as f:
        f.write(PROMPT + "\n")
    with open(os.path.join(handoff_dir, "README.txt"), "w") as f:
        f.write(instructions(handoff_dir, reference_copy) + "\n")
    return reference_copy


def instructions(handoff_dir, reference_copy):
    return "\n".join([
        f"Gemini hand-off folder: {handoff_dir}",
        "  1. Open the Gemini app (https://gemini.google.com) and select image generation (Nano Banana).",
        "  2. Upload these two images, in this order:",
        f"       (1) {os.path.join(handoff_dir, '1_grey_garment.png')}   grey garment render (green background)",
        f"       (2) {reference_copy}   reference photo",
        f"  3. Send this prompt (also in {os.path.join(handoff_dir, 'PROMPT.txt')}):",
        f"       {PROMPT}",
        "  4. Check that the texture matches the reference and the garment kept its shape and pose; regenerate in Gemini",
        "     if needed. Download the image and save it (any file name, .png/.jpg/.webp) into:",
        f"       {os.path.join(handoff_dir, 'result')}/",
    ])


def find_result(handoff_dir):
    """Newest image in <handoff_dir>/result/, or None."""
    images = [p for p in glob.glob(os.path.join(handoff_dir, "result", "*")) if p.lower().endswith(IMAGE_EXTS)]
    return max(images, key=os.path.getmtime) if images else None


def import_result(result, grey, output, min_iou=0.85):
    """Chroma-key the Gemini image, scale and shift its garment onto the grey render's silhouette, and paste it onto
    the render's green background (Gemini often draws the garment larger or slightly shifted)."""
    from omnifabric.utils.fit_view import fit_scaled_view, green_screen_mask

    render = np.asarray(Image.open(grey).convert("RGB"))
    r, g, b = render.astype(np.int16).transpose(2, 0, 1)
    silhouette = ~((g > r + 30) & (g > b + 30))  # the render's own, exact green screen
    render_rgba = np.dstack([render, np.where(silhouette, 255, 0).astype(np.uint8)])

    generated = np.asarray(Image.open(result).convert("RGB"))
    fitted, scale, iou_before, iou_after = fit_scaled_view(render_rgba, generated)
    out = np.where(silhouette[..., None], fitted[..., :3], render)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    Image.fromarray(out).save(output)
    print(f"  Imported {result}\n  -> {output}")
    print(f"  Garment fitted to the render: scale {scale:.3f}, silhouette IoU {iou_before:.3f} -> {iou_after:.3f}")
    fg = green_screen_mask(generated, erode=0)
    if fg[0].any() or fg[-1].any() or fg[:, 0].any() or fg[:, -1].any():
        print("  Warning: the garment touches the border of Gemini's image, so part of it may be cut off; check "
              f"{output}, and regenerate in Gemini (or use FLUX.2) if needed.")
    if iou_after < min_iou:
        print(f"  Warning: Gemini's garment shape differs from the grey garment (IoU {iou_after:.2f}); check "
              f"{output}, and regenerate in Gemini (or use FLUX.2) if the texture does not fit the garment.")


def try_import(result, grey, output):
    """Import a result image; False (with the reason printed) if it cannot be read."""
    try:
        import_result(result, grey, output)
    except OSError as e:
        print(f"\n  Could not read {result} ({e}). Delete it and save Gemini's image again, or choose another option.")
        return False
    return True


def ask(question, options, default=None):
    """Keep asking until one of the options' keys is entered; Ctrl-D quits."""
    keys = [key for key, _ in options]
    while True:
        print(question)
        for key, text in options:
            print(f"    [{key}] {text}")
        print("  Choice: ", end="", flush=True)
        line = sys.stdin.readline()
        if not line:  # EOF
            print()
            return "q"
        answer = line.strip().lower() or default
        if answer in keys:
            return answer
        print(f"  Please enter one of: {', '.join(keys)}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grey", required=True, help="frontal garment-only render on a green background")
    parser.add_argument("--reference", required=True, help="the input photo")
    parser.add_argument("--handoff_dir", required=True, help="folder for the Gemini inputs, prompt and result")
    parser.add_argument("--output", required=True, help="textured frontal view to write (views/front_garment.png)")
    parser.add_argument("--mode", choices=["ask", "gemini", "flux2"], default="ask")
    parser.add_argument("--garment_name", default="", help="shown in the menu")
    args = parser.parse_args()

    if args.mode == "flux2":
        return 0

    if args.mode == "ask":
        print(f"\n---------------- Textured frontal view {args.garment_name} ----------------")
        choice = ask("How should the garment's frontal texture be generated?", [
            ("1", "Gemini (Nano Banana): manual, you generate the image in the Gemini app (as in the paper)"),
            ("2", "FLUX.2-dev: automatic, runs locally"),
        ], default="1")
        if choice == "q":
            return WAITING
        if choice == "2":
            return 0

    reference_copy = prepare_handoff(args.grey, args.reference, args.handoff_dir)
    result = find_result(args.handoff_dir)
    if result and try_import(result, args.grey, args.output):
        return 0

    print("\n" + instructions(args.handoff_dir, reference_copy))
    if args.mode == "gemini":  # non-interactive: stop here, the next run imports the image
        print("\nThen re-run the same command; finished steps are reused.")
        return WAITING

    while True:
        choice = ask("\nWhen the image is saved:", [
            ("c", f"Continue: I saved Gemini's image into {os.path.join(args.handoff_dir, 'result')}/"),
            ("f", "Use FLUX.2 instead"),
            ("q", "Quit for now: re-run the same command later (finished steps are reused)"),
        ], default="c")
        if choice == "f":
            return 0
        if choice == "q":
            print(f"  Stopped. Save Gemini's image into {os.path.join(args.handoff_dir, 'result')}/ and re-run the "
                  f"same command (TEXTURE_BACKEND=gemini skips the menu).")
            return WAITING
        result = find_result(args.handoff_dir)
        if result is None:
            print(f"\n  No image found in {os.path.join(args.handoff_dir, 'result')}/ yet. Please (re)do these steps:\n")
            print(instructions(args.handoff_dir, reference_copy))
        elif try_import(result, args.grey, args.output):
            return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
