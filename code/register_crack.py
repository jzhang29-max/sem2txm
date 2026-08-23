"""Register a SEM/TXM pair on the CRACK, not on texture.

Texture-based mutual information registers these pairs (the overlays show the same
crack, unambiguously) but cannot PIN the scale: on B3 the NMI runs 1.0245-1.0269
across ratios 1.9 to 2.9, a flat curve, so peak-picking on it is noise. Two
modalities that share almost no texture statistics will do that.

What they do share is one large, distinctive, high-contrast feature -- the crack.
Its outline is the same physical object in both images, so aligning the crack MASKS
is far better conditioned than aligning texture: the crack's length and width both
scale with the ratio, which makes scale identifiable rather than flat.

Method: segment the crack in each image as the largest dark elongated component,
then search scale / rotation / translation for the transform that maximises Dice
overlap between the two masks. Reports the Dice curve, so a flat one is visible as
such instead of being reported as a peak.
"""
import argparse
import json

import numpy as np
from scipy import ndimage as ndi

import config as C


def crack_mask(img01, dark_pct=8.0, min_frac=0.002, elong_min=2.0):
    """Largest dark elongated component. img01 is flat-fielded, in [0,1]."""
    valid = img01 > 1e-6
    if valid.sum() < 100:
        return None
    thr = np.percentile(img01[valid], dark_pct)
    m = (img01 <= thr) & valid
    m = ndi.binary_opening(m, iterations=2)
    lab, n = ndi.label(m)
    if n == 0:
        return None
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    for i in order[:12]:
        if sizes[i] < min_frac * img01.size:
            break
        comp = lab == i
        ys, xs = np.nonzero(comp)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        if max(h, w) / max(min(h, w), 1) >= elong_min:
            return comp
    return (lab == order[0]) if sizes[order[0]] > 0 else None


def dice(a, b):
    inter = np.logical_and(a, b).sum()
    tot = a.sum() + b.sum()
    return float(2.0 * inter / tot) if tot else 0.0


def best_overlap(src_mask, dst_mask, ratio, angle):
    """Scale+rotate src, then find the translation maximising Dice via FFT."""
    from skimage.transform import rescale, rotate as skrotate
    s = rescale(src_mask.astype(np.float32), 1.0 / ratio, anti_aliasing=False,
                order=0, preserve_range=True) > 0.5
    if angle:
        s = skrotate(s.astype(np.float32), angle, resize=False, order=0,
                     preserve_range=True) > 0.5
    if s.shape[0] >= dst_mask.shape[0] or s.shape[1] >= dst_mask.shape[1]:
        return None
    if s.sum() < 50:
        return None
    # Cross-correlate the two binary masks; the peak is the best translation.
    from scipy.signal import fftconvolve
    corr = fftconvolve(dst_mask.astype(np.float32),
                       s[::-1, ::-1].astype(np.float32), mode="valid")
    iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
    win = dst_mask[iy:iy + s.shape[0], ix:ix + s.shape[1]]
    if win.shape != s.shape:
        return None
    return dice(s, win), (int(iy), int(ix)), s.shape


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sem", required=True)
    ap.add_argument("--txm", required=True)
    ap.add_argument("--sem-um-per-px", type=float, default=0.0)
    ap.add_argument("--ratios", default="1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.2,3.5,4.0")
    ap.add_argument("--angles", default="-6,-4,-2,0,2,4,6")
    ap.add_argument("--out", default="")
    ap.add_argument("--overlay", default="")
    args = ap.parse_args()

    from run_pairs import load_and_prep
    from read_scale import from_metadata
    sem, _, _ = load_and_prep(args.sem, "sem")
    txm, _, _ = load_and_prep(args.txm, "txm")
    sem_um = args.sem_um_per_px or (from_metadata(args.sem)[0] or 0.0)

    ms = crack_mask(sem)
    mt = crack_mask(txm)
    if ms is None or mt is None:
        raise SystemExit("could not segment a crack in one of the images")
    print(f"SEM {sem.shape}  crack {int(ms.sum())} px ({ms.mean():.2%})")
    print(f"TXM {txm.shape}  crack {int(mt.sum())} px ({mt.mean():.2%})")
    if sem_um:
        print(f"SEM scale {sem_um:.6f} um/px")
    print()

    rows = []
    print(f"{'ratio':>6s} {'angle':>6s} {'Dice':>8s} {'implied TXM um/px':>19s}")
    for r in (float(x) for x in args.ratios.split(",")):
        for a in (float(x) for x in args.angles.split(",")):
            got = best_overlap(ms, mt, r, a)
            if got is None:
                continue
            d, at, shp = got
            rows.append({"ratio": r, "angle": a, "dice": round(d, 4),
                         "at": at, "template": list(shp)})
    if not rows:
        raise SystemExit("no usable transform")
    rows.sort(key=lambda z: -z["dice"])
    best = rows[0]
    # Print the best angle per ratio, so the scale curve is legible.
    byr = {}
    for z in rows:
        byr.setdefault(z["ratio"], z)
    for r in sorted(byr):
        z = byr[r]
        star = " <" if z is best else ""
        um = f"{r * sem_um:19.4f}" if sem_um else f"{'':>19s}"
        print(f"{r:6.2f} {z['angle']:6.1f} {z['dice']:8.4f} {um}{star}")

    dv = np.array([byr[r]["dice"] for r in sorted(byr)])
    spread = float(dv.max() - dv.min())
    print(f"\nbest: ratio {best['ratio']} angle {best['angle']} Dice {best['dice']:.4f}")
    print(f"Dice spread across ratios: {spread:.4f}")
    if best["dice"] < 0.2:
        print("VERDICT: overlap too low to call this a registration.")
    elif spread < 0.05:
        print("VERDICT: the curve is flat -- the crack constrains position but not")
        print("  scale. Do not quote a ratio from it.")
    else:
        print(f"VERDICT: scale is identifiable. ratio {best['ratio']}")
        if sem_um:
            print(f"  ==> TXM pixel size = {best['ratio'] * sem_um:.4f} um/px")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"best": best, "curve": [byr[r] for r in sorted(byr)],
                       "sem_um_per_px": sem_um, "dice_spread": spread}, f, indent=1)
    if args.overlay:
        write_mask_overlay(ms, mt, best, args.overlay)
        print(f"  overlay -> {args.overlay}")


def write_mask_overlay(ms, mt, best, path):
    from PIL import Image
    from skimage.transform import rescale, rotate as skrotate
    s = rescale(ms.astype(np.float32), 1.0 / best["ratio"], anti_aliasing=False,
                order=0, preserve_range=True) > 0.5
    if best["angle"]:
        s = skrotate(s.astype(np.float32), best["angle"], resize=False, order=0,
                     preserve_range=True) > 0.5
    y, x = best["at"]
    win = mt[y:y + s.shape[0], x:x + s.shape[1]]
    h, w = win.shape
    s = s[:h, :w]
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = s * 255          # SEM crack in red
    rgb[..., 1] = win * 255        # TXM crack in green
    Image.fromarray(rgb).save(path)   # overlap appears yellow


if __name__ == "__main__":
    main()
