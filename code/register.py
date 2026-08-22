"""Try to register a SEM frame onto a TXM mosaic, and say how well it worked.

Why this is not a normal registration problem. The two images are of different
physical quantities -- SEM is backscatter off a surface, TXM is a line integral
through the specimen -- so intensities are not related by any monotone function
and plain cross-correlation of grey values is meaningless. What IS shared is
structure: a crack that reaches the surface appears in both.

So the search runs on gradient magnitude (structure, not intensity) to localise,
and scores candidates with normalised mutual information (which asks only whether
one image predicts the other, not whether they look alike).

The unknown scale is the point of the exercise. Neither repo records um/px for TXM,
so the SEM:TXM pixel ratio is unknown, and that ratio is the most likely reason the
translator is training on mismatched fields of view. A confident registration at
ratio r says the SEM covers r x fewer microns per pixel than the TXM.

Output is deliberately sceptical: a ratio is only reported alongside its NMI and
the margin over the next-best candidate, because a template match always returns
SOMETHING and a peak that barely beats its neighbours is not a registration.
"""
import argparse
import json

import numpy as np
from scipy import ndimage as ndi

import config as C


def structure(img, sigma=2.0):
    """Gradient magnitude, contrast-normalised. Modality-independent-ish."""
    g = ndi.gaussian_gradient_magnitude(np.asarray(img, np.float32), sigma)
    lo, hi = np.percentile(g, [1, 99])
    return np.clip((g - lo) / max(hi - lo, 1e-8), 0, 1)


def nmi(a, b, bins=48):
    """Normalised mutual information, 2 = identical, ~1 = independent."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    h, _, _ = np.histogram2d(a, b, bins=bins)
    p = h / max(h.sum(), 1e-12)
    px, py = p.sum(1), p.sum(0)
    def H(q):
        q = q[q > 0]
        return -(q * np.log(q)).sum()
    hx, hy = H(px), H(py)
    hxy = H(p.ravel())
    if hxy <= 0:
        return 0.0
    return float((hx + hy) / hxy)


def best_match(sem01, txm01, ratio, angle=0.0, top_k=5):
    """Downsample SEM by `ratio` (and rotate by `angle`) to TXM pixel size, then
    locate it in the TXM.

    Rotation is searched because "same region, not aligned" allows the specimen to
    have been remounted between instruments. Returns
    (nmi, (y, x), template_shape, ncc) or None if the template cannot fit.
    """
    from skimage.feature import match_template
    from skimage.transform import rescale, rotate as skrotate
    t = rescale(sem01, 1.0 / ratio, anti_aliasing=True, preserve_range=True)
    if angle:
        # Rotate then trim the invalid border the rotation introduces.
        t = skrotate(t, angle, resize=False, mode="reflect", preserve_range=True)
        cut = int(np.ceil(abs(np.sin(np.deg2rad(angle))) * max(t.shape))) + 2
        if 2 * cut < min(t.shape):
            t = t[cut:-cut, cut:-cut]
    if t.shape[0] >= txm01.shape[0] or t.shape[1] >= txm01.shape[1]:
        return None
    if min(t.shape) < 32:
        return None
    ts, xs = structure(t), structure(txm01)
    corr = match_template(xs, ts, pad_input=False)
    flat = np.argsort(corr.ravel())[::-1][:top_k]
    best = None
    for idx in flat:
        y, x = np.unravel_index(idx, corr.shape)
        win = txm01[y:y + t.shape[0], x:x + t.shape[1]]
        if win.shape != t.shape:
            continue
        score = nmi(t, win)
        if best is None or score > best[0]:
            best = (score, (int(y), int(x)), t.shape, float(corr[y, x]))
    return best


# Measured independence floor. Two unrelated random fields score NMI 1.004, and a
# real dry run -- SEM 260622_316_H_b2_front_CBS_01 against the b2 TXM frames, same
# specimen -- scored 1.0005 to 1.006 at every ratio tried, with margins of 0.0008.
# So anything at or below about 1.01 is noise, and this is the number a claimed
# registration has to clear. It exists because a template match always returns
# SOMETHING.
NMI_FLOOR = 1.01
MIN_MARGIN = 0.01


def register(sem_stem, txm_stem, ratios, angles=(0.0,), verbose=True):
    sem = np.load(C.CACHE / "sem" / f"{sem_stem}.npy").astype(np.float32) / 255.0
    txm = np.load(C.CACHE / "txm" / f"{txm_stem}.npy").astype(np.float32) / 255.0
    rows = []
    for r in ratios:
        for a in angles:
            m = best_match(sem, txm, r, a)
            if m is None:
                continue
            rows.append({"ratio": float(r), "angle": float(a), "nmi": round(m[0], 4),
                         "at": m[1], "template": list(m[2]), "ncc": round(m[3], 4)})
            if verbose:
                print(f"    ratio {r:5.2f}  angle {a:+5.1f}  NMI {m[0]:.4f}  "
                      f"NCC {m[3]:+.3f}  template {m[2]} at {m[1]}")
    if not rows:
        return None
    rows.sort(key=lambda d: -d["nmi"])
    best, second = rows[0], (rows[1] if len(rows) > 1 else None)
    best["margin_over_next"] = round(best["nmi"] - second["nmi"], 4) if second else None
    best["all"] = rows
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sem", required=True, help="SEM stem")
    ap.add_argument("--txm", required=True, help="TXM stem")
    ap.add_argument("--ratios", default="1.5,2,3,4,5,6,7,8,10,12",
                    help="SEM-pixels-per-TXM-pixel candidates to try")
    ap.add_argument("--angles", default="0",
                    help="rotations in degrees to try, e.g. -6,-3,0,3,6")
    ap.add_argument("--overlay", default="",
                    help="write a checkerboard overlay of the best match here")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    ratios = [float(x) for x in args.ratios.split(",")]
    angles = [float(x) for x in args.angles.split(",")]
    print(f"{args.sem}  ->  {args.txm}")
    best = register(args.sem, args.txm, ratios, angles)
    if best is None:
        print("  no candidate ratio produced a usable template")
        return
    print(f"\n  best: ratio {best['ratio']}  NMI {best['nmi']}  "
          f"margin over next {best['margin_over_next']}")
    if best["nmi"] < NMI_FLOOR or (best["margin_over_next"] or 0) < MIN_MARGIN:
        print("  VERDICT: not a registration. NMI is near the independence floor "
              "(~1.0)\n           or the peak does not beat its neighbours. Treat the "
              "ratio as unknown.")
    else:
        print("  VERDICT: candidate registration -- inspect the overlay before "
              "trusting it.")
    if args.overlay:
        write_overlay(args.sem, args.txm, best, args.overlay)
        print(f"  overlay -> {args.overlay}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(best, f, indent=1)


def write_overlay(sem_stem, txm_stem, best, path, tile=64):
    """Checkerboard the registered SEM against the TXM it landed on.

    A number cannot tell you a registration is real; a checkerboard can. If the
    structures continue across the tile boundaries it is aligned, and if they jump
    at every edge it is not.
    """
    from PIL import Image
    from skimage.transform import rescale, rotate as skrotate
    sem = np.load(C.CACHE / "sem" / f"{sem_stem}.npy").astype(np.float32) / 255.0
    txm = np.load(C.CACHE / "txm" / f"{txm_stem}.npy").astype(np.float32) / 255.0
    t = rescale(sem, 1.0 / best["ratio"], anti_aliasing=True, preserve_range=True)
    if best.get("angle"):
        t = skrotate(t, best["angle"], resize=False, mode="reflect", preserve_range=True)
        cut = int(np.ceil(abs(np.sin(np.deg2rad(best["angle"]))) * max(t.shape))) + 2
        if 2 * cut < min(t.shape):
            t = t[cut:-cut, cut:-cut]
    y, x = best["at"]
    h, w = t.shape
    win = txm[y:y + h, x:x + w]
    if win.shape != t.shape:
        return
    def norm(a):
        lo, hi = np.percentile(a, [1, 99])
        return np.clip((a - lo) / max(hi - lo, 1e-8), 0, 1)
    a, b = norm(t), norm(win)
    yy, xx = np.mgrid[:h, :w]
    check = (((yy // tile) + (xx // tile)) % 2).astype(bool)
    mix = np.where(check, a, b)
    sheet = np.concatenate([a, b, mix], axis=1)
    Image.fromarray((sheet * 255).astype(np.uint8)).save(path)


if __name__ == "__main__":
    main()
