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


# Margin inside a window within which a pixel's features are indistinguishable
# from its whole-frame features.
#
# Set by measurement, not by eyeballing the filter list. The first guess was 32 px
# -- reasoning from GRADIENT/LAPLACIAN_SIGMAS, which stop at 8 -- and it left a
# max error of 1.8e-2 in three channels. Those channels are smooth_s16/32/64:
# SMOOTH_SIGMAS runs to 64, and a Gaussian needs ~3 sigma to die out.
#
# At 192 px the same comparison leaves 1.5e-4, in smooth_s64 alone -- not exact,
# but two orders of magnitude down and immaterial to a tree ensemble on features
# in [0,1]. Window must exceed 2 * margin to leave any interior, hence 768, which
# also runs 4.5x faster per usable pixel than a whole-frame pass on a 2.9 MP
# mosaic and far better than that on the 23 MP one.
FILTER_MARGIN = 192
WINDOW = 768


def reflect_pad(img01):
    """Pad by FILTER_MARGIN with reflection so a window core can sit anywhere in
    the original frame, borders included.

    Without this, cores can only cover the frame minus a 192 px rim, which biased
    the test prevalence badly: the four GT frames read 36.7 / 43.5 / 47.1% crack
    windowed against 25.5 / 27.0 / 29.7% whole-frame, because the cracks are
    central and the excluded rim is mostly not-crack. IoU against a wrong
    prevalence is a wrong IoU.

    Reflection is the right padding rather than a convenience: compute_feature_stack
    filters with scipy's default mode='reflect', so a reflect-padded window
    reproduces what a whole-frame pass computes at the border.
    """
    m = FILTER_MARGIN
    return np.pad(np.asarray(img01, np.float32), m, mode="reflect")


def core_tiles(h, w):
    """Tile the frame with non-overlapping cores. Every pixel is in exactly one.

    Random sliding origins do NOT sample a frame uniformly: with a core of 384,
    a pixel 5 rows from the edge lies in 6 possible windows while a central one
    lies in 384, so the centre is over-represented ~64x. That is what made the
    test prevalence read 42% against a true 25%. A tiling has no such weighting.
    Returns (core, [(y, x, h_i, w_i), ...]).
    """
    core = WINDOW - 2 * FILTER_MARGIN
    tiles = []
    for y in range(0, h, core):
        for x in range(0, w, core):
            tiles.append((y, x, min(core, h - y), min(core, w - x)))
    return core, tiles


def sample_windows(img01, pos_mask, neg_mask, n, rng, win=WINDOW, max_windows=400):
    """Rows from random windows rather than from a whole-frame feature stack.

    Computing compute_feature_stack() on a full mosaic is ~18 Gaussian passes over
    up to 23 megapixels, and the experiment needs it once per image PER SEED --
    hundreds of full-frame passes. Windows give identical features for any pixel
    at least FILTER_MARGIN inside the window (the filters are local), at a small
    fraction of the cost. Pixels in the margin are discarded rather than used with
    edge-contaminated context.
    """
    h, w = img01.shape
    if min(h, w) < 64:
        return None
    padded = reflect_pad(img01)
    core, tiles = core_tiles(h, w)
    rng.shuffle(tiles)
    tiles = tiles[:max_windows]
    Xs, yl = [], []
    got_pos = got_neg = 0
    want_pos = n // 2
    want_neg = n - want_pos
    for (cy, cx, ch, cw) in tiles:
        if got_pos >= want_pos and got_neg >= want_neg:
            break
        pm = pos_mask[cy:cy + ch, cx:cx + cw]
        nm = neg_mask[cy:cy + ch, cx:cx + cw]
        if not pm.any() and not nm.any():
            continue
        wnd = padded[cy:cy + ch + 2 * FILTER_MARGIN, cx:cx + cw + 2 * FILTER_MARGIN]
        st = compute_feature_stack(wnd)
        m = FILTER_MARGIN
        flat = st[m:m + ch, m:m + cw, :].reshape(-1, st.shape[-1])
        for mask, want, lab in ((pm, want_pos, 1), (nm, want_neg, 0)):
            got = got_pos if lab == 1 else got_neg
            need = want - got
            if need <= 0 or not mask.any():
                continue
            idx = np.flatnonzero(mask.ravel())
            take = idx if len(idx) <= need else rng.choice(idx, need, replace=False)
            Xs.append(flat[take])
            yl.append(np.full(len(take), lab, np.int8))
            if lab == 1:
                got_pos += len(take)
            else:
                got_neg += len(take)
    if not Xs:
        return None
    return np.concatenate(Xs), np.concatenate(yl)


def sample_from(img01, pos_mask, neg_mask, n, rng):
    """Balanced rows for one image, or None if either class is absent here."""
    if n <= 0 or not pos_mask.any() or not neg_mask.any():
        return None
    return sample_windows(img01, pos_mask, neg_mask, n, rng)


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


def sem_negative_mask(stem, m, far_px):
    """Negatives for one SEM frame, cached on disk.

    The expensive part is the Euclidean distance transform used to find unmarked
    pixels far from anything the reviewer touched -- on a 25 megapixel frame that
    is seconds, and it was being recomputed for every frame, every sampler call,
    every seed: 162 times over a 3-seed run. It depends only on the mask, so it
    is computed once per frame and kept.
    """
    cdir = C.CACHE / "sem_neg"
    cdir.mkdir(parents=True, exist_ok=True)
    cf = cdir / f"{stem}_far{far_px}.npz"
    if cf.exists():
        z = np.load(cf)
        far = np.unpackbits(z["far"], count=m.size).reshape(m.shape).astype(bool)
        return (m == 2) | far, int(z["n_explicit"]), int(z["n_far"])
    touched = m > 0
    far = (~touched) & (ndi.distance_transform_edt(~touched) > far_px)
    np.savez_compressed(cf, far=np.packbits(far),
                        n_explicit=int((m == 2).sum()), n_far=int(far.sum()))
    return (m == 2) | far, int((m == 2).sum()), int(far.sum())


SEM_MAX_FRAMES = 0   # 0 = use every eligible frame; set to pin the frame set


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
    # Arms B and C must draw on the SAME frames, or C wins or loses on frame count
    # rather than on whether the translation helped. So the eligible set is the
    # intersection: a mask, a preprocessed frame, AND a translation on disk --
    # regardless of which of the two this call is sampling.
    have = []
    for e in man["sem"]:
        mp = C.SEM_MASK_DIR / f"{e['stem']}_correction_mask.png"
        if (mp.exists()
                and (C.CACHE / "sem" / f"{e['stem']}.npy").exists()
                and (C.CACHE / "translated" / f"{e['stem']}.npy").exists()):
            have.append((e["stem"], mp))
    # Pinning the frame set makes two runs comparable. Between the iteration-1000
    # and iteration-5000 runs BOTH the translator and the number of translated
    # frames changed (17 -> 37), so arm C moved even though it never touches the
    # translator, and neither change could be attributed. Capping here isolates one
    # variable at a time. Deterministic: `have` is in sorted stem order, which is
    # the same order translate.py writes in.
    if SEM_MAX_FRAMES:
        have = have[:SEM_MAX_FRAMES]
    rng.shuffle(have)
    per = max(1, budget // max(len(have), 1))
    Xs, ys = [], []
    got = 0
    n_explicit = n_far = 0
    used = set()
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
        neg, n_e, n_f = sem_negative_mask(stem, m, far_px)
        n_explicit += n_e; n_far += n_f
        r = sample_from(img01, crack, neg, min(per, budget - got), rng)
        if r is None:
            continue
        Xs.append(r[0]); ys.append(r[1]); got += len(r[1]); used.add(stem)
    if not Xs:
        return None, None, {}
    return (np.concatenate(Xs), np.concatenate(ys),
            {"n_frames_eligible": len(have), "n_frames_used": len(used),
             "frames": sorted(used),
             "explicit_negatives": n_explicit, "far_negatives": n_far})


def build_test_cached(per_frame=120000, seed=12345):
    """build_test() is deterministic given its seed but takes about ten minutes,
    most of it the 170 tiles of the 23.5 MP frame. Cache it so reruns and repeated
    experiments do not pay for it again."""
    cf = C.CACHE / f"test_set_p{per_frame}_s{seed}.npz"
    if cf.exists():
        z = np.load(cf, allow_pickle=True)
        stems = [str(x) for x in z["stems"]]
        out = {s: (z[f"X_{s}"], z[f"y_{s}"]) for s in stems}
        print(f"test set from cache ({len(out)} frames, "
              f"{sum(len(v[1]) for v in out.values())} rows)")
        for s, (_, y) in out.items():
            print(f"  test {s[:38]:38s} n={len(y):6d} pos={y.mean():.1%}")
        return out
    out = build_test(np.random.default_rng(seed), per_frame)
    if out:
        d = {"stems": np.array(list(out.keys()))}
        for k, (X, y) in out.items():
            d[f"X_{k}"] = X
            d[f"y_{k}"] = y
        np.savez(cf, **d)
        print(f"test set cached -> {cf.name}")
    return out


def build_test(rng, per_frame=120000):
    """Test rows from the four dense-GT frames, at their NATURAL prevalence.

    Deliberately unlike the training sampler in one way: it does not balance the
    classes. Prevalence is what makes AUC and IoU mean anything on frames that run
    18.5-29.7% crack, and the sampled prevalence is checked against the whole-frame
    value below so a sampling bias cannot pass silently.
    """
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
        h, w = img01.shape
        padded = reflect_pad(img01)
        core, tiles = core_tiles(h, w)
        m = FILTER_MARGIN
        total_area = sum(t[2] * t[3] for t in tiles)
        Xs, ys = [], []
        for (cy, cx, ch, cw) in tiles:
            # Allocate in proportion to tile area so edge tiles, which are
            # smaller, are neither over- nor under-weighted.
            want = int(round(per_frame * (ch * cw) / total_area))
            if want < 1:
                continue
            wnd = padded[cy:cy + ch + 2 * m, cx:cx + cw + 2 * m]
            st = compute_feature_stack(wnd)
            flat = st[m:m + ch, m:m + cw, :].reshape(-1, st.shape[-1])
            lab = g[cy:cy + ch, cx:cx + cw].reshape(-1)
            k = min(want, len(lab))
            idx = rng.choice(len(lab), k, replace=False)
            Xs.append(flat[idx]); ys.append(lab[idx].astype(np.int8))
        if not Xs:
            print(f"  {stem}: no usable windows")
            continue
        X = np.concatenate(Xs); Y = np.concatenate(ys)
        out[stem] = (X, Y)
        truth = float(g.mean())
        flag = "" if abs(Y.mean() - truth) < 0.03 else "   <-- SAMPLING BIAS"
        print(f"  test {stem[:38]:38s} n={len(Y):6d} "
              f"pos={Y.mean():.1%} (whole frame {truth:.1%}){flag}")
    return out


def equalise(arms, rng):
    """Give every arm the same number of positives and the same number of negatives.

    The row budget alone does not achieve this. Arm A draws its whole budget from
    real TXM and can fall short when an image has no labels of one class -- in the
    first run A got 43,632 rows against 50,114 for B and C, so B and C had 15% more
    data than the baseline they were being compared to. Any difference measured
    that way is partly a difference in dataset size.

    Trimming to the common minimum costs rows but makes the comparison mean what it
    claims to mean.
    """
    if not arms:
        return arms, {}
    counts = {k: (int((y == 1).sum()), int((y == 0).sum())) for k, (X, y) in arms.items()}
    n_pos = min(c[0] for c in counts.values())
    n_neg = min(c[1] for c in counts.values())
    out = {}
    for k, (X, y) in arms.items():
        pi = np.flatnonzero(y == 1)
        ni = np.flatnonzero(y == 0)
        keep = np.concatenate([rng.choice(pi, n_pos, replace=False),
                               rng.choice(ni, n_neg, replace=False)])
        rng.shuffle(keep)
        out[k] = (X[keep], y[keep])
    return out, {"before": counts, "equalised_to": [n_pos, n_neg]}


def fit_eval(Xtr, ytr, test, seed):
    """AUC is the headline because it needs no threshold. IoU is reported at the
    threshold that maximises it ON THE TEST FRAMES, which is an optimistic ceiling
    rather than a deployable number -- every arm gets the same favour, so the
    comparison between arms is still fair, but the absolute value is not
    comparable to a deployed model's IoU."""
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
    ap.add_argument("--sem-max-frames", type=int, default=0,
                    help="use only the first N eligible SEM frames, so a rerun can "
                         "hold the frame set fixed while the translator changes")
    ap.add_argument("--out", default=str(C.OUT / "label_transfer.json"))
    args = ap.parse_args()

    global SEM_MAX_FRAMES
    SEM_MAX_FRAMES = args.sem_max_frames
    results = {"row_budget": args.rows, "seeds": args.seeds,
               "sem_max_frames": args.sem_max_frames, "arms": {},
               "notes": {"iou_caveat": ("iou_at_best uses the threshold that "
                                        "maximises IoU on the test frames; it is an "
                                        "optimistic ceiling, applied equally to "
                                        "every arm. AUC is threshold-free.")}}
    print("building test set from the four dense-GT frames ...")
    test = build_test_cached()
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

        arms, eq = equalise(arms, np.random.default_rng(seed + 900))
        if eq:
            print(f"  equalised every arm to {eq['equalised_to'][0]} positives + "
                  f"{eq['equalised_to'][1]} negatives (was {eq['before']})")
            results["notes"]["equalisation"] = eq
        for name, (X, y) in arms.items():
            per = fit_eval(X, y, test, seed)
            mean_auc = float(np.mean([v["auc"] for v in per.values()]))
            mean_iou = float(np.mean([v["iou_at_best"] for v in per.values()]))
            results["arms"].setdefault(name, []).append(
                {"seed": seed, "n_rows": int(len(y)), "pos_frac": round(float(y.mean()), 3),
                 "mean_auc": round(mean_auc, 4), "mean_iou": round(mean_iou, 4),
                 "per_frame": per})
            print(f"  {name:32s} n={len(y):7d} AUC {mean_auc:.4f}  "
                  f"IoU* {mean_iou:.4f}")

    print("\n================ summary (mean over seeds) ================")
    print("  AUC is threshold-free. IoU* is at the best threshold on the test")
    print("  frames -- an optimistic ceiling, equal favour to every arm.")
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
