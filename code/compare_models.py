"""Put several checkpoints side by side against a real TXM crop, with numbers.

Two things decide whether a translation is any use here, and they pull against
each other, so neither can be read alone:

  affine_r2   how much of the output is just the input rescaled. High means the
              model changed contrast and called it a modality transfer -- the
              failure the project owner spotted by eye before any metric named it.
  crack_r     whether hand-marked cracks keep their local contrast. Low means the
              output may look different but has lost the structure that made it
              worth predicting.

A useful model needs affine_r2 low AND crack_r high. The point of this script is
that you can watch a hyperparameter trade one for the other.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import config as C


def affine_r2(x, y):
    xr, yr = np.asarray(x, np.float64).ravel(), np.asarray(y, np.float64).ravel()
    xc, yc = xr - xr.mean(), yr - yr.mean()
    d = (xc * xc).sum()
    if d <= 0:
        return float("nan")
    a = (xc * yc).sum() / d
    res = yc - a * xc
    tot = (yc * yc).sum()
    return float(1 - (res ** 2).sum() / tot) if tot > 0 else float("nan")


def norm(z):
    lo, hi = np.percentile(z, [1, 99])
    return np.clip((z - lo) / max(hi - lo, 1e-8), 0, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="paths, or name=path pairs for nicer labels")
    ap.add_argument("--sem", default="260622_316_H_b2_front_CBS_01")
    ap.add_argument("--txm-ref", default="260618_b2_338_13")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--at", default="1700,2200", help="y,x of the SEM crop")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=str(C.FIGURES / "model_comparison.png"))
    args = ap.parse_args()

    from train import device_of
    from translate import load_generator, translate
    from eval_translation import crack_contrast

    dev = device_of(args.device)
    S = args.size
    y0, x0 = (int(v) for v in args.at.split(","))
    x = np.load(C.CACHE / "sem" / f"{args.sem}.npy").astype(np.float32) / 255.0
    mp = C.SEM_MASK_DIR / f"{args.sem}_correction_mask.png"
    mask = None
    if mp.exists():
        with Image.open(mp) as im:
            m = np.array(im)
        if m.shape == x.shape:
            mask = (m[y0:y0 + S, x0:x0 + S] == 1)
            print(f"crack mask in crop: {int(mask.sum())} px")
        else:
            print(f"mask shape {m.shape} != image {x.shape}; crack_r unavailable")
    else:
        print(f"no correction mask for {args.sem}; crack_r unavailable")
    x = x[y0:y0 + S, x0:x0 + S]

    panels = [("SEM input", x)]
    rows = []
    for spec in args.ckpts:
        # rpartition, not partition: labels legitimately contain "=" (a run is
        # named by the hyperparameter it changed, e.g. "nce=0.25"), and splitting
        # on the FIRST one made the path start mid-label.
        label, _, path = spec.rpartition("=")
        if not path:
            path, label = label, Path(label).parent.name
        p = Path(path)
        if not p.is_absolute():
            p = C.ROOT / p
        if not p.exists():
            print(f"  missing: {p}")
            continue
        G, ck = load_generator(p, dev)
        y = translate(G, x, dev, 512, 256, 4, offsets=2)
        # affine_r2 over the whole crop AND over sub-patches: it is strongly
        # region-dependent (0.32 on one field, 0.68 on another for the same model),
        # so a single number from a single crop is not a property of the model.
        ar2 = affine_r2(x, y)
        sub = []
        for yy in range(0, S - 255, 256):
            for xx in range(0, S - 255, 256):
                sub.append(affine_r2(x[yy:yy + 256, xx:xx + 256],
                                     y[yy:yy + 256, xx:xx + 256]))
        sub = [v for v in sub if v == v]
        cr = None
        if mask is not None and mask.sum() > 500:
            b = crack_contrast(x, mask, rng=np.random.default_rng(0))
            a = crack_contrast(y, mask, rng=np.random.default_rng(0))
            k = min(len(b), len(a))
            if k > 2:
                cr = float(np.corrcoef([t[0] for t in b][:k],
                                       [t[0] for t in a][:k])[0, 1])
        rows.append({"label": label, "iter": ck["iter"], "affine_r2": round(ar2, 4),
                     "affine_r2_patch_mean": round(float(np.mean(sub)), 4) if sub else None,
                     "affine_r2_patch_sd": round(float(np.std(sub)), 4) if sub else None,
                     "crack_r": None if cr is None else round(cr, 4),
                     "std": round(float(y.std()), 4)})
        panels.append((f"{label} (it {ck['iter']})", y))

    t = np.load(C.CACHE / "txm" / f"{args.txm_ref}.npy").astype(np.float32) / 255.0
    th, tw = t.shape
    panels.append(("REAL TXM (reference)",
                   t[(th - S) // 2:(th - S) // 2 + S, (tw - S) // 2:(tw - S) // 2 + S]))

    W, head, gap = S // 2, 26, 6
    sheet = Image.new("L", (len(panels) * W + (len(panels) - 1) * gap, W + head), 255)
    d = ImageDraw.Draw(sheet)
    for i, (lab, z) in enumerate(panels):
        im = Image.fromarray((norm(z) * 255).astype(np.uint8)).resize((W, W),
                                                                     Image.LANCZOS)
        sheet.paste(im, (i * (W + gap), head))
        d.text((i * (W + gap) + 3, 8), lab[:34], fill=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out}\n")

    print(f"{'model':22s} {'affR2':>7s} {'affR2 per-patch':>17s} {'crack_r':>9s} "
          f"{'std':>8s}")
    print(f"{'(want)':22s} {'LOW':>7s} {'LOW':>17s} {'HIGH':>9s}")
    for r in rows:
        cr = "n/a" if r["crack_r"] is None else f"{r['crack_r']:.4f}"
        pm = ("n/a" if r["affine_r2_patch_mean"] is None
              else f"{r['affine_r2_patch_mean']:.3f}+-{r['affine_r2_patch_sd']:.3f}")
        print(f"{r['label'][:22]:22s} {r['affine_r2']:7.4f} {pm:>17s} {cr:>9s} "
              f"{r['std']:8.4f}")
    print(f"{'SEM input':30s} {'--':>10s} {'--':>9s} {x.std():8.4f}")
    print(f"{'real TXM':30s} {'--':>10s} {'--':>9s} {panels[-1][1].std():8.4f}")


if __name__ == "__main__":
    main()
