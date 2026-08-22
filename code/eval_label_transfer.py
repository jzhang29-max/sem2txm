"""Does a SEM label, carried through the translator, teach a TXM crack detector?

This is the falsifiable claim, and it needs three controls to mean anything:

  A  real TXM only              what you can do today, without any SEM
  B  real TXM + TRANSLATED SEM  the proposal
  C  real TXM + RAW SEM         the control that matters most
  D  translated SEM only        how far you get with no TXM crack labels at all

C is the one that can kill the idea. If pasting untranslated SEM frames into the
training set helps as much as translated ones, then the translator is doing no
work and the gain was only ever "more crack examples from somewhere". B must beat
C, not just A.

Every arm gets the SAME total row budget, so no arm can win by having more data --
for B and C the budget is split half TXM, half SEM. Test frames are the four dense
hand-drawn GT frames, which the translator never saw (excluded by specimen in
prep.py) and which contribute no training rows to any arm.

Honest about the inputs: the real-TXM positive labels are rule-derived
(write_positive_crack_labels.py -- dark-relative-to-local, elongated,
inside-specimen), not hand-drawn, so arm A is a rule-taught baseline. The SEM
positives ARE hand-drawn. That asymmetry favours B and is stated rather than
hidden.
"""
import argparse
import json
import sys

import numpy as np
from scipy import ndimage as ndi

import config as C

sys.path.insert(0, str(C.TXM_REPO / "code"))
from txm_features import compute_feature_stack, N_FEATURES  # noqa: E402

TEST_FRAMES = {
    "260618_B2_333_75_um_zoom": "333_75_um_zoom_gt",
    "260618_b2_336_25": "336_25_gt",
    "260618_b2_338_13": "338_13_gt",
    "260618_b2_343_75_LARGE": "LARGE_343_75_gt",
}


def load_gt():
    z = np.load(C.TXM_REPO / "dataset_cache" / "masks.npz", allow_pickle=True)
    return {k: z[k] for k in z.keys()}


def feature_rows(img01, ys, xs):
    """Feature stack is computed on the WHOLE frame (the filters are global) and
    then sampled, so a sampled pixel sees the same context it would at inference."""
    st = compute_feature_stack(np.asarray(img01, np.float32))
    return st[ys, xs, :]


def sample_from(img01, pos_mask, neg_mask, n, rng):
    pos = np.flatnonzero(pos_mask.ravel())
    neg = np.flatnonzero(neg_mask.ravel())
    if len(pos) == 0 or len(neg) == 0:
        return None
    npos = min(len(pos), n // 2)
    nneg = min(len(neg), n - npos)
    pi = rng.choice(pos, npos, replace=False)
    ni = rng.choice(neg, nneg, replace=False)
    idx = np.concatenate([pi, ni])
    y, x = np.unravel_index(idx, img01.shape)
    X = feature_rows(img01, y, x)
    yy = np.concatenate([np.ones(npos, np.int8), np.zeros(nneg, np.int8)])
    return X, yy


def txm_rows(budget, rng, exclude_stems):
    """Rows from the real TXM mosaics, using the app's own crack / not-crack labels."""
    lab = np.load(C.TXM_REPO / "paint" / "app_labels.npz", allow_pickle=True)
    man = json.loads((C.CACHE / "manifest.json").read_text())
    usable = [e for e in man["txm"] if not e["reference"] and e["stem"] not in exclude_stems]
    keymap = {}
    for k in lab.keys():
        for e in usable:
            if e["stem"] in k:
                keymap[e["stem"]] = k
    stems = [e["stem"] for e in usable if e["stem"] in keymap]
    rng.shuffle(stems)
    per = max(1, budget // max(len(stems), 1))
    Xs, ys = [], []
    got = 0
    for s in stems:
        if got >= budget:
            break
        img01 = np.load(C.CACHE / "txm" / f"{s}.npy").astype(np.float32) / 255.0
        m = lab[keymap[s]]
        if m.shape != img01.shape:
            continue
        r = sample_from(img01, m == 1, m == 2, min(per, budget - got), rng)
        if r is None:
            continue
        Xs.append(r[0]); ys.append(r[1]); got += len(r[1])
    return (np.concatenate(Xs), np.concatenate(ys)) if Xs else (None, None)


def sem_rows(budget, rng, translated, far_px=50):
    """Rows from the SEM frames that carry a hand-drawn correction mask.

    Negatives: pixels explicitly hand-marked not-crack, plus unmarked pixels
    further than `far_px` from anything the reviewer touched. Unmarked pixels are
    NOT negatives in general -- in this dataset unmarked means "the model's
    opinion stands" -- so distance from any mark is used as the qualifier, and
    the count of each kind is reported.
    """
    from PIL import Image
    man = json.loads((C.CACHE / "manifest.json").read_text())
    src = (C.CACHE / "translated") if translated else (C.CACHE / "sem")
    have = []
    for e in man["sem"]:
        mp = C.SEM_MASK_DIR / f"{e['stem']}_correction_mask.png"
        if mp.exists() and (src / f"{e['stem']}.npy").exists():
            have.append((e["stem"], mp))
    rng.shuffle(have)
    per = max(1, budget // max(len(have), 1))
    Xs, ys = [], []
    got = 0
    n_explicit = n_far = 0
    for stem, mp in have:
        if got >= budget:
            break
        img01 = np.load(src / f"{stem}.npy").astype(np.float32) / 255.0
        with Image.open(mp) as im:
            m = np.array(im)
        if m.shape != img01.shape:
            continue
        crack = m == 1
        if crack.sum() < 500:
            continue
        touched = m > 0
        far = (~touched) & (ndi.distance_transform_edt(~touched) > far_px)
        neg = (m == 2) | far
        n_explicit += int((m == 2).sum()); n_far += int(far.sum())
        r = sample_from(img01, crack, neg, min(per, budget - got), rng)
        if r is None:
            continue
        Xs.append(r[0]); ys.append(r[1]); got += len(r[1])
    if not Xs:
        return None, None, {}
    return (np.concatenate(Xs), np.concatenate(ys),
            {"n_frames": len(have), "explicit_negatives": n_explicit, "far_negatives": n_far})


def build_test(rng, per_frame=120000):
    gt = load_gt()
    out = {}
    for stem, key in TEST_FRAMES.items():
        p = C.CACHE / "txm" / f"{stem}.npy"
        if not p.exists() or key not in gt:
            print(f"  test frame missing: {stem}")
            continue
        img01 = np.load(p).astype(np.float32) / 255.0
        g = gt[key]
        if g.shape != img01.shape:
            print(f"  SHAPE MISMATCH {stem}: img {img01.shape} vs gt {g.shape}")
            continue
        n = min(per_frame, img01.size)
        idx = rng.choice(img01.size, n, replace=False)
        y, x = np.unravel_index(idx, img01.shape)
        out[stem] = (feature_rows(img01, y, x), g[y, x].astype(np.int8))
        print(f"  test {stem[:40]:40s} {img01.shape} pos={out[stem][1].mean():.1%}")
    return out


def fit_eval(Xtr, ytr, test, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                         random_state=seed)
    clf.fit(Xtr, ytr)
    per = {}
    for stem, (Xte, yte) in test.items():
        p = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p)
        best = max(((2 * ((p > t) & (yte == 1)).sum() /
                     max(((p > t).sum() + (yte == 1).sum()), 1)), t)
                   for t in np.linspace(0.1, 0.9, 17))
        # Dice at the best threshold -> IoU
        dice = best[0]
        per[stem] = {"auc": round(float(auc), 4),
                     "iou_at_best": round(float(dice / (2 - dice)), 4)}
    return per


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=160000, help="row budget per arm")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--out", default=str(C.OUT / "label_transfer.json"))
    args = ap.parse_args()

    results = {"row_budget": args.rows, "seeds": args.seeds, "arms": {}, "notes": {}}
    print("building test set from the four dense-GT frames ...")
    test = build_test(np.random.default_rng(12345))
    if not test:
        print("no test frames available; aborting")
        return

    want = [a.strip() for a in args.arms.split(",")]
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        half = args.rows // 2
        cache = {}
        print(f"\n=== seed {seed} ===")
        if any(a in want for a in ("A", "B", "C")):
            print("  sampling real TXM rows ...")
            cache["txm_full"] = txm_rows(args.rows, np.random.default_rng(seed), set())
            cache["txm_half"] = txm_rows(half, np.random.default_rng(seed + 100), set())
        if any(a in want for a in ("B", "D")):
            print("  sampling translated SEM rows ...")
            xt, yt, meta = sem_rows(half if "B" in want else args.rows,
                                    np.random.default_rng(seed + 200), translated=True)
            cache["sem_t_half"] = (xt, yt)
            xt2, yt2, _ = sem_rows(args.rows, np.random.default_rng(seed + 300),
                                   translated=True)
            cache["sem_t_full"] = (xt2, yt2)
            results["notes"]["sem_negatives"] = meta
        if "C" in want:
            print("  sampling raw SEM rows ...")
            xr, yr, _ = sem_rows(half, np.random.default_rng(seed + 200), translated=False)
            cache["sem_r_half"] = (xr, yr)

        arms = {}
        if "A" in want and cache.get("txm_full", (None,))[0] is not None:
            arms["A_real_txm_only"] = cache["txm_full"]
        if "B" in want and cache.get("sem_t_half", (None,))[0] is not None:
            arms["B_txm_plus_translated_sem"] = (
                np.concatenate([cache["txm_half"][0], cache["sem_t_half"][0]]),
                np.concatenate([cache["txm_half"][1], cache["sem_t_half"][1]]))
        if "C" in want and cache.get("sem_r_half", (None,))[0] is not None:
            arms["C_txm_plus_raw_sem"] = (
                np.concatenate([cache["txm_half"][0], cache["sem_r_half"][0]]),
                np.concatenate([cache["txm_half"][1], cache["sem_r_half"][1]]))
        if "D" in want and cache.get("sem_t_full", (None,))[0] is not None:
            arms["D_translated_sem_only"] = cache["sem_t_full"]

        for name, (X, y) in arms.items():
            per = fit_eval(X, y, test, seed)
            mean_auc = float(np.mean([v["auc"] for v in per.values()]))
            mean_iou = float(np.mean([v["iou_at_best"] for v in per.values()]))
            results["arms"].setdefault(name, []).append(
                {"seed": seed, "n_rows": int(len(y)), "pos_frac": round(float(y.mean()), 3),
                 "mean_auc": round(mean_auc, 4), "mean_iou": round(mean_iou, 4),
                 "per_frame": per})
            print(f"  {name:32s} n={len(y):7d} AUC {mean_auc:.4f}  IoU {mean_iou:.4f}")

    print("\n================ summary (mean over seeds) ================")
    for name, runs in results["arms"].items():
        a = np.array([r["mean_auc"] for r in runs])
        i = np.array([r["mean_iou"] for r in runs])
        print(f"  {name:32s} AUC {a.mean():.4f} +-{a.std():.4f}   "
              f"IoU {i.mean():.4f} +-{i.std():.4f}")
        results["arms"][name] = {"runs": runs,
                                 "auc_mean": round(float(a.mean()), 4),
                                 "auc_sd": round(float(a.std()), 4),
                                 "iou_mean": round(float(i.mean()), 4),
                                 "iou_sd": round(float(i.std()), 4)}
    C.OUT.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
