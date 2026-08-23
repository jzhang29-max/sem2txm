"""Estimate the SEM:TXM pixel ratio from texture statistics, with no pairs.

The unknown scale ratio is the flaw most likely to invalidate everything else in
this repo: training runs at 1:1 pixels, and if a SEM pixel covers 7x less material
than a TXM pixel then the generator has been asked to match a 259 um field against
a 1800 um one. Two attempts to recover the number from the data failed (registration
across modalities, and the stage positions in the filenames), and it cannot be read
off the TXM files.

But it can be ESTIMATED without any registration, because there is one ratio at
which the two domains should look most alike. Downsample SEM by r, and ask a
classifier to tell the result from real TXM. If the modalities are genuinely
comparable at some physical scale, separability should DIP at the r matching that
scale, because at every other r the classifier can win on texture scale alone.

This is an estimate, not a calibration -- it assumes the two modalities share
texture statistics at the correct scale, which is an assumption about the physics
and not something this measures. Reported with the curve, so a flat curve (no
preferred scale) is visible as such rather than being reported as a minimum.
"""
import argparse
import json

import numpy as np

import config as C
from eval_translation import descriptors, DESC_NAMES


def separability(sem_patches, txm_patches, sem_src, txm_src, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    X = np.concatenate([np.stack([descriptors(p) for p in sem_patches]),
                        np.stack([descriptors(p) for p in txm_patches])])
    y = np.concatenate([np.zeros(len(sem_patches)), np.ones(len(txm_patches))])
    groups = np.array([f"s:{s}" for s in sem_src] + [f"t:{s}" for s in txm_src])
    aucs = []
    for tr, te in GroupKFold(n_splits=4).split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=150, random_state=seed)
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratios", default="1,1.5,2,3,4,6,8,10")
    ap.add_argument("--n", type=int, default=700, help="patches per domain")
    ap.add_argument("--patch", type=int, default=128,
                    help="comparison size AFTER downsampling, so every ratio is "
                         "judged on the same number of pixels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(C.OUT / "scale_match.json"))
    args = ap.parse_args()

    from skimage.transform import rescale
    rng = np.random.default_rng(args.seed)
    txm = np.load(C.CACHE / "bank_txm.npy", mmap_mode="r")
    tsrc = [s[0] for s in json.loads((C.CACHE / "bank_txm_src.json").read_text())]
    # SEM comes from the FULL cached frames, not the 256 px bank: testing a ratio
    # of r needs a crop of P*r pixels, so the bank caps the search at r=2 and the
    # interesting range is 4-10.
    sem_frames = sorted(p for p in (C.CACHE / "sem").glob("*.npy")
                        if not p.name.endswith(".box.npy"))
    print(f"{len(sem_frames)} SEM frames available for cropping")

    ti = np.sort(rng.choice(len(txm), args.n, replace=False))
    # TXM is the reference: centre-crop it to the comparison size.
    P = args.patch
    def centre(a):
        h, w = a.shape
        y, x = (h - P) // 2, (w - P) // 2
        return np.asarray(a[y:y + P, x:x + P], np.float32) / 255.0
    txm_p = [centre(txm[i]) for i in ti]
    txm_s = [tsrc[i] for i in ti]

    ratios = [float(x) for x in args.ratios.split(",")]
    rows = []
    print(f"comparing {args.n} patches per domain at {P}x{P} after downsampling\n")
    print(f"{'ratio':>7s} {'SEM px used':>12s} {'C2ST AUC':>10s} {'sd':>7s}")
    for r in ratios:
        need = int(np.ceil(P * r))
        sem_p, sem_s = [], []
        tries = 0
        while len(sem_p) < args.n and tries < args.n * 40:
            tries += 1
            fp = sem_frames[int(rng.integers(0, len(sem_frames)))]
            a = np.load(fp, mmap_mode="r")
            h, w = a.shape
            if h <= need or w <= need:
                continue
            y = int(rng.integers(0, h - need))
            x = int(rng.integers(0, w - need))
            crop = np.asarray(a[y:y + need, x:x + need], np.float32) / 255.0
            if r != 1.0:
                crop = rescale(crop, 1.0 / r, anti_aliasing=True, preserve_range=True)
            crop = np.asarray(crop[:P, :P], np.float32)
            if crop.shape != (P, P):
                continue
            sem_p.append(crop)
            sem_s.append(fp.stem)
        if len(sem_p) < args.n // 2:
            print(f"{r:7.2f} {need:12d}   too few frames large enough")
            continue
        auc, sd = separability(sem_p, txm_p, sem_s, txm_s, args.seed)
        rows.append({"ratio": r, "sem_px": need, "auc": round(auc, 4),
                     "sd": round(sd, 4), "n": len(sem_p)})
        print(f"{r:7.2f} {need:12d} {auc:10.4f} {sd:7.4f}")

    if len(rows) < 3:
        print("\nnot enough points to read a curve")
        return
    aucs = np.array([d["auc"] for d in rows])
    best = rows[int(np.argmin(aucs))]
    spread = float(aucs.max() - aucs.min())
    print(f"\nminimum separability at ratio {best['ratio']} (AUC {best['auc']})")
    print(f"curve spans {aucs.min():.4f} to {aucs.max():.4f}  (spread {spread:.4f})")
    if spread < 0.02:
        print("\nVERDICT: the curve is flat. No scale makes the two domains")
        print("  meaningfully more alike, so this does NOT estimate a ratio -- and")
        print("  it also means scale mismatch is not what the classifier is using")
        print("  to separate them.")
    elif best["ratio"] == min(d["ratio"] for d in rows):
        print("\nVERDICT: minimum sits at the edge of the range searched. Widen it")
        print("  before reading anything into the location.")
    else:
        print(f"\nVERDICT: a preferred scale exists near {best['ratio']}x.")
        print(f"  Read as: one TXM pixel covers about {best['ratio']}x the material")
        print(f"  of one SEM pixel. An ESTIMATE resting on the assumption that the")
        print(f"  modalities share texture statistics at the correct scale.")
    C.OUT.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "min_ratio": best["ratio"],
                   "spread": round(spread, 4)}, f, indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
