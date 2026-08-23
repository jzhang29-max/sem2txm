"""Refine a SEM->TXM registration by direct optimisation, and report the residual.

The grid search in register.py locates the pair (the overlays confirm the same
crack) but cannot resolve scale: mutual information runs 1.0245-1.0269 across
ratios 1.9-2.9 on B3, a flat curve. And ORB/RANSAC refinement fails outright across
these modalities -- the descriptors do not correspond.

So refine by optimising NMI continuously over (scale, rotation, ty, tx) from the
grid solution, on a smoothed multi-resolution pyramid. Coarse levels have far fewer
local optima; each level's answer initialises the next.

The residual is then estimated by BLOCK CONSISTENCY rather than by feature matching:
divide the overlap into blocks, register each block independently over a small
window, and take the spread of those local shifts. If the global transform were
exact, every block would report ~zero shift. The spread is what a paired loss has to
tolerate, and it is the number that decides which loss is legitimate.
"""
import argparse
import json

import numpy as np
from scipy import ndimage as ndi

import config as C


def warp(sem, scale, angle, ty, tx, out_shape):
    """Resample SEM into TXM space: divide by `scale`, rotate, translate."""
    from skimage.transform import SimilarityTransform, warp as skwarp
    # Map output (TXM) coords -> input (SEM) coords.
    tf = (SimilarityTransform(translation=(-tx, -ty))
          + SimilarityTransform(rotation=np.deg2rad(angle))
          + SimilarityTransform(scale=scale))
    return skwarp(sem, tf, output_shape=out_shape, order=1, mode="constant",
                  cval=np.nan, preserve_range=True)


def masked_nmi(a, b, bins=48):
    m = np.isfinite(a) & np.isfinite(b) & (b > 1e-6)
    if m.sum() < 5000:
        return 0.0
    from register import nmi
    return nmi(a[m], b[m], bins)


def optimise(sem, txm, x0, levels=(8, 4, 2), verbose=True):
    from scipy.optimize import minimize
    best = list(x0)
    for lv in levels:
        s = ndi.zoom(sem, 1.0 / lv, order=1)
        t = ndi.zoom(txm, 1.0 / lv, order=1)
        def neg(p):
            sc, ang, ty, tx = p
            if not (0.2 < sc < 20):
                return 0.0
            w = warp(s, sc, ang, ty / lv, tx / lv, t.shape)
            return -masked_nmi(w, t)
        p0 = np.array(best, float)
        r = minimize(neg, p0, method="Powell",
                     options={"xtol": 1e-3, "ftol": 1e-4, "maxiter": 2000})
        best = list(r.x)
        if verbose:
            print(f"    level 1/{lv}: scale {best[0]:.4f} angle {best[1]:+.3f} "
                  f"shift ({best[2]:+.1f},{best[3]:+.1f})  NMI {-r.fun:.4f}")
    return best, -r.fun


def block_residual(sem_w, txm, block=256, search=48, min_valid=0.6):
    """Local shift per block; the spread is the residual the transform leaves."""
    from skimage.registration import phase_cross_correlation
    H, W = txm.shape
    shifts = []
    for y in range(0, H - block, block):
        for x in range(0, W - block, block):
            a = sem_w[y:y + block, x:x + block]
            b = txm[y:y + block, x:x + block]
            m = np.isfinite(a) & (b > 1e-6)
            if m.mean() < min_valid:
                continue
            aa = np.where(np.isfinite(a), a, np.nanmean(a))
            sh, _, _ = phase_cross_correlation(b, aa, upsample_factor=4,
                                               normalization=None)
            if max(abs(sh[0]), abs(sh[1])) <= search:
                shifts.append(sh)
    if len(shifts) < 4:
        return None
    sh = np.array(shifts)
    mag = np.hypot(sh[:, 0], sh[:, 1])
    return {"n_blocks": len(sh),
            "median_px": round(float(np.median(mag)), 2),
            "p90_px": round(float(np.percentile(mag, 90)), 2),
            "iqr_px": round(float(np.percentile(mag, 75) - np.percentile(mag, 25)), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=str(C.ROOT / "pairs.json"))
    ap.add_argument("--report", default=str(C.OUT / "pairs" / "pairs_report.json"))
    ap.add_argument("--out", default=str(C.OUT / "pairs" / "refined.json"))
    args = ap.parse_args()

    from run_pairs import load_and_prep
    from pathlib import Path
    spec = json.loads(Path(args.pairs).read_text())
    prev = {p["name"]: p for p in json.loads(Path(args.report).read_text())["pairs"]}
    out = {}
    for pr in spec["pairs"]:
        name = pr["name"]
        if name not in prev:
            continue
        print(f"=== {name}")
        sem, _, _ = load_and_prep(pr["sem"], "sem")
        txm, _, _ = load_and_prep(pr["txm"], "txm")
        c = prev[name]["coarse"]
        y, x = c["at"]
        x0 = [c["ratio"], c.get("angle", 0.0), float(y), float(x)]
        print(f"    start: scale {x0[0]} angle {x0[1]} at ({y},{x})")
        best, score = optimise(sem, txm, x0)
        w = warp(sem, best[0], best[1], best[2], best[3], txm.shape)
        res = block_residual(w, txm)
        cov = float(np.isfinite(w).mean())
        print(f"    final NMI {score:.4f}   SEM covers {cov:.1%} of the TXM frame")
        if res:
            print(f"    block residual: median {res['median_px']} px, "
                  f"p90 {res['p90_px']} px, over {res['n_blocks']} blocks")
        else:
            print("    block residual: too few valid blocks to estimate")
        out[name] = {"scale": best[0], "angle": best[1], "ty": best[2], "tx": best[3],
                     "nmi": round(score, 4), "coverage": round(cov, 4),
                     "residual": res,
                     "sem_um_per_px": prev[name].get("sem_um_per_px"),
                     "txm_um_per_px_implied": (
                         round(best[0] * prev[name]["sem_um_per_px"], 5)
                         if prev[name].get("sem_um_per_px") else None)}
        if out[name]["txm_um_per_px_implied"]:
            print(f"    implies TXM {out[name]['txm_um_per_px_implied']} um/px")
        print()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
