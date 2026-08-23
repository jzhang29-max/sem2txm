"""SEM image in, predicted TXM image out.

    ./run predict my_micrograph.tif
    ./run predict folder_of_sems/ --out results/

Does the whole chain on an arbitrary file, so nothing has to be in the cache first:
read -> crop the instrument info panel -> pseudo-flat-field -> normalise -> tiled
translation -> write. Accepts .tif .tiff .png .jpg .jpeg .bmp, 8- or 16-bit, grey
or RGB, any size.

WHAT THE OUTPUT IS. A re-rendering of the surface you gave it, in the appearance of
a destitched and flat-fielded TXM mosaic, with its geometry held in place. It is
NOT a measurement of the specimen interior. SEM sees a surface; TXM is a line
integral through the bulk; nothing in a surface image determines what lies beneath
it. A feature in the output is a restatement of a feature in your input, not
evidence of anything at depth.

Also worth knowing before trusting it (all measured, see the README): the outputs
are still distinguishable from real TXM by a classifier at AUC 0.87, and adding
these predictions to a TXM crack detector's training set did not reliably improve
it. Use this to look at and to reason with, not as a substitute for a TXM
measurement.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

import config as C

sys.path.insert(0, str(C.TXM_REPO / "code"))
sys.path.insert(0, str(C.SEM_REPO / "code"))

EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

# Default to the run with the pixel-space identity loss: it holds held-out identity
# pearson 0.871 against 0.718, and a two-sample AUC of 0.869 against 0.967. Both
# better than the original run; crack-contrast correlation is slightly worse
# (0.868 vs 0.921), which matters for label transfer but not for looking at.
DEFAULT_CKPTS = ["runs/cut_idtfix/final.pt", "runs/cut_idtfix/ckpt.pt",
                 "runs/cut/final.pt", "runs/cut/ckpt.pt"]


def pick_checkpoint(given=""):
    if given:
        p = Path(given)
        if not p.exists():
            raise SystemExit(f"checkpoint not found: {p}")
        return p
    for rel in DEFAULT_CKPTS:
        p = C.ROOT / rel
        if p.exists():
            return p
    raise SystemExit("no checkpoint found; train one with ./run train")


def read_image(path):
    """Any of the accepted formats to a 2-D float array, without rescaling."""
    p = Path(path)
    if p.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        a = tifffile.imread(str(p))
    else:
        from PIL import Image
        with Image.open(p) as im:
            a = np.array(im)
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2) if a.shape[-1] >= 3 else a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"{p.name}: expected a 2-D image, got shape {a.shape}")
    return a.astype(np.float32)


def preprocess(raw, crop_panel=True, sigma=24.0):
    """Same chain prep.py uses, so a prediction matches what the model trained on."""
    import flatfield
    from txm_features import robust_normalize
    box = None
    if crop_panel:
        from prep import sem_crop_box
        try:
            box = sem_crop_box(raw)
            x0, y0, x1, y1 = box
            if (y1 - y0) >= 64 and (x1 - x0) >= 64:
                raw = raw[y0:y1, x0:x1]
            else:
                box = None
        except Exception:
            box = None
    ff = flatfield.flatfield(raw, sigma_y=sigma, sigma_x=sigma)
    if isinstance(ff, tuple):
        ff = ff[0]
    return np.asarray(robust_normalize(np.asarray(ff, np.float64), 1.0, 99.0),
                      np.float32), box


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="SEM image files, or folders of them")
    ap.add_argument("--out", default=str(C.OUT / "predictions"))
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=256,
                    help="tile overlap; 256 measured lowest tiling artifact")
    ap.add_argument("--offsets", type=int, default=2, choices=(1, 2, 4),
                    help="average over N shifted tile grids. InstanceNorm makes a "
                         "tile's output brightness depend on that tile's content, "
                         "so a single grid leaves soft rectangular patches worth "
                         "16%% of the image std; 2 grids halves that for 2x the "
                         "time.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-crop-panel", action="store_true",
                    help="skip info-panel detection (use if already cropped)")
    ap.add_argument("--no-side-by-side", action="store_true")
    ap.add_argument("--png-scale", type=int, default=1,
                    help="downsample factor for the PNGs; 1 = full resolution")
    args = ap.parse_args()

    files = []
    for item in args.inputs:
        p = Path(item)
        if p.is_dir():
            files += [f for f in sorted(p.iterdir()) if f.suffix.lower() in EXTS]
        elif p.suffix.lower() in EXTS:
            files.append(p)
        else:
            print(f"skipping {p} (not one of {', '.join(EXTS)})")
    if not files:
        raise SystemExit("no readable images found")

    from PIL import Image
    import tifffile
    from train import device_of
    from translate import load_generator, translate

    ckpt = pick_checkpoint(args.ckpt)
    dev = device_of(args.device)
    G, ck = load_generator(ckpt, dev)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"model {ckpt.relative_to(C.ROOT)} (iter {ck['iter']}) on {dev}")
    print(f"{len(files)} image(s) -> {outdir}\n")

    for i, f in enumerate(files, 1):
        try:
            raw = read_image(f)
        except Exception as e:
            print(f"  [{i}/{len(files)}] {f.name}: unreadable ({e})")
            continue
        img01, box = preprocess(raw, crop_panel=not args.no_crop_panel)
        pred = translate(G, img01, dev, args.tile, args.overlap, args.batch,
                         offsets=args.offsets)

        stem = f.stem
        # float TIFF for analysis, PNG for looking at.
        tifffile.imwrite(str(outdir / f"{stem}_predicted_txm.tif"),
                         pred.astype(np.float32))

        def to_png(a):
            im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
            if args.png_scale > 1:
                im = im.resize((max(1, im.width // args.png_scale),
                                max(1, im.height // args.png_scale)),
                               Image.LANCZOS)
            return im
        to_png(pred).save(outdir / f"{stem}_predicted_txm.png")
        if not args.no_side_by_side:
            a, b = np.array(to_png(img01)), np.array(to_png(pred))
            gap = 8
            sheet = np.full((a.shape[0], a.shape[1] + gap + b.shape[1]), 255, np.uint8)
            sheet[:, :a.shape[1]] = a
            sheet[:, a.shape[1] + gap:] = b
            Image.fromarray(sheet).save(outdir / f"{stem}_side_by_side.png")

        crop_note = ""
        if box is not None and (box[3] - box[1], box[2] - box[0]) != raw.shape:
            crop_note = f"  (panel cropped: {raw.shape} -> {img01.shape})"
        print(f"  [{i}/{len(files)}] {f.name}  {img01.shape}{crop_note}")

    print(f"\nwrote *_predicted_txm.tif (float), *_predicted_txm.png, "
          f"*_side_by_side.png")
    print("Reminder: this re-renders your SURFACE in TXM appearance. It does not "
          "see\nbeneath it, and it is still distinguishable from real TXM "
          "(AUC 0.87).")


if __name__ == "__main__":
    main()
