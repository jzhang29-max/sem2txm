"""Does the output look like TXM, and is it still the same picture?

Four measurements, because "the samples look good" is not a result:

1. TWO-SAMPLE TEST. Fit a classifier to tell real TXM patches from translated
   ones using interpretable descriptors (intensity moments, multi-scale local
   std, banded power spectrum). AUC 0.5 means the two sets are indistinguishable
   to that classifier; 1.0 means it separates them perfectly. Reported with the
   descriptor that carries the separation, because knowing WHICH statistic gives
   the model away is more useful than the scalar.

2. POWER SPECTRUM. Radially averaged, for source SEM / translated / real TXM.
   This is where the effective-resolution gap between the modalities shows up,
   and it is the honest answer to "are these two even at the same scale".

3. INTENSITY STATS against the real TXM distribution.

4. CRACK CONTRAST RETENTION -- the one that matters for label transfer. For every
   hand-marked crack region in the SEM set, measure its contrast against a local
   ring of background, on the SEM input and again on the translated output. If a
   crack is still darker than its surroundings after translation, and by a
   proportional amount, a mask drawn on the SEM still describes the output. This
   is measured LOCALLY, against a ring, specifically to avoid having to treat
   unreviewed pixels as negatives -- which they are not.
"""
import argparse
import json

import numpy as np
import torch
from scipy import ndimage as ndi

import config as C


# ------------------------------------------------------------------ descriptors

def radial_power(p, nbands=8):
    """Radially averaged power spectrum in log-spaced bands, normalised to sum 1
    so that overall contrast does not leak into the shape of the spectrum."""
    f = np.fft.fftshift(np.abs(np.fft.fft2(p - p.mean())) ** 2)
    h, w = f.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    rmax = min(h, w) / 2
    edges = np.geomspace(2, rmax, nbands + 1)
    out = []
    for i in range(nbands):
        m = (r >= edges[i]) & (r < edges[i + 1])
        out.append(f[m].mean() if m.any() else 0.0)
    out = np.asarray(out)
    return out / max(out.sum(), 1e-12)


def descriptors(p):
    """p: float32 patch in [0,1]."""
    g = np.hypot(*np.gradient(p))
    d = [p.mean(), p.std()]
    d += list(np.percentile(p, [1, 25, 50, 75, 99]))
    for s in (2, 8):
        m = ndi.gaussian_filter(p, s)
        m2 = ndi.gaussian_filter(p.astype(np.float64) ** 2, s)
        d.append(float(np.sqrt(np.clip(m2 - m.astype(np.float64) ** 2, 0, None)).mean()))
    d.append(g.mean())
    d += list(radial_power(p))
    return np.asarray(d, np.float32)


DESC_NAMES = (["mean", "std", "p1", "p25", "p50", "p75", "p99",
               "localstd_s2", "localstd_s8", "gradmag"]
              + [f"power_band{i}" for i in range(8)])


# ------------------------------------------------------------------ 1. C2ST

def two_sample_test(real, fake, real_src, fake_src, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    X = np.concatenate([np.stack([descriptors(p) for p in real]),
                        np.stack([descriptors(p) for p in fake])])
    y = np.concatenate([np.zeros(len(real)), np.ones(len(fake))])
    # Group by SOURCE IMAGE so the test measures distribution separation, not
    # memorisation of one frame's quirks.
    groups = np.array([f"r:{s}" for s in real_src] + [f"f:{s}" for s in fake_src])

    aucs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))

    # Which single descriptor separates them best, on its own.
    singles = []
    for j, nm in enumerate(DESC_NAMES):
        a = roc_auc_score(y, X[:, j])
        singles.append((nm, max(a, 1 - a)))
    singles.sort(key=lambda t: -t[1])
    return float(np.mean(aucs)), float(np.std(aucs)), singles[:5]


# ------------------------------------------------------------------ 4. contrast

def crack_contrast(img, mask, min_area=200, ring=6, max_regions=400, rng=None):
    """Per-region (inside mean - ring mean). Negative == crack is darker."""
    lab, n = ndi.label(mask)
    if n == 0:
        return []
    # bincount over the label image, and one find_objects pass -- both O(image)
    # rather than O(image * regions), which matters at 25 megapixels.
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    objs = ndi.find_objects(lab)
    ids = [i for i in range(1, n + 1) if sizes[i] >= min_area]
    if rng is not None and len(ids) > max_regions:
        ids = list(rng.choice(ids, max_regions, replace=False))
    out = []
    for i in ids:
        sl = objs[i - 1]
        if sl is None:
            continue
        pad = tuple(slice(max(0, s.start - ring), min(d, s.stop + ring))
                    for s, d in zip(sl, lab.shape))
        sub_lab = lab[pad] == i
        sub_img = img[pad]
        inside = sub_img[sub_lab]
        ringmask = ndi.binary_dilation(sub_lab, iterations=ring) & ~sub_lab
        outside = sub_img[ringmask]
        if len(inside) < 20 or len(outside) < 20:
            continue
        out.append((float(inside.mean() - outside.mean()), int(sub_lab.sum())))
    return out


# ------------------------------------------------------------------ driver

@torch.no_grad()
def translate_patches(G, patches, dev, batch=16):
    out = []
    for i in range(0, len(patches), batch):
        t = torch.from_numpy(patches[i:i + batch][:, None] * 2 - 1).to(dev)
        o = G(t).cpu().numpy()[:, 0]
        out.append((o + 1) * 0.5)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(C.ROOT / "runs" / "cut" / "ckpt.pt"))
    ap.add_argument("--n-patches", type=int, default=1200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(C.OUT / "eval_translation.json"))
    args = ap.parse_args()

    from train import device_of
    from translate import load_generator
    dev = device_of(args.device)
    G, ck = load_generator(args.ckpt, dev)
    rng = np.random.default_rng(args.seed)
    res = {"ckpt_iter": ck["iter"]}

    sem = np.load(C.CACHE / "bank_sem.npy", mmap_mode="r")
    txm = np.load(C.CACHE / "bank_txm.npy", mmap_mode="r")
    sem_src = [s[0] for s in json.loads((C.CACHE / "bank_sem_src.json").read_text())]
    txm_src = [s[0] for s in json.loads((C.CACHE / "bank_txm_src.json").read_text())]

    n = args.n_patches
    si = rng.choice(len(sem), min(n, len(sem)), replace=False)
    ti = rng.choice(len(txm), min(n, len(txm)), replace=False)
    sem_p = np.asarray(sem[np.sort(si)], np.float32) / 255.0
    txm_p = np.asarray(txm[np.sort(ti)], np.float32) / 255.0
    sem_s = [sem_src[i] for i in np.sort(si)]
    txm_s = [txm_src[i] for i in np.sort(ti)]

    print(f"translating {len(sem_p)} SEM patches ...")
    fake_p = translate_patches(G, sem_p, dev)

    print("1. two-sample test (real TXM vs translated) ...")
    auc, sd, singles = two_sample_test(txm_p, fake_p, txm_s, sem_s, args.seed)
    res["c2st_auc"] = round(auc, 4)
    res["c2st_auc_sd"] = round(sd, 4)
    res["c2st_top_descriptors"] = [[nm, round(a, 4)] for nm, a in singles]
    print(f"   AUC {auc:.3f} +-{sd:.3f}  (0.5 = indistinguishable)")
    for nm, a in singles:
        print(f"     {nm:16s} alone: {a:.3f}")

    print("2. power spectra ...")
    spec = {
        "sem_source": radial_power(sem_p[0] * 0 + sem_p.mean(0)).tolist(),
        "sem_mean": np.stack([radial_power(p) for p in sem_p[:300]]).mean(0).tolist(),
        "translated_mean": np.stack([radial_power(p) for p in fake_p[:300]]).mean(0).tolist(),
        "txm_mean": np.stack([radial_power(p) for p in txm_p[:300]]).mean(0).tolist(),
    }
    spec.pop("sem_source")
    res["power_spectrum_bands"] = {k: [round(v, 5) for v in x] for k, x in spec.items()}

    print("3. intensity stats ...")
    def st(a):
        return {"median": round(float(np.median(a)), 4),
                "iqr": round(float(np.percentile(a, 75) - np.percentile(a, 25)), 4),
                "std": round(float(a.std()), 4)}
    res["intensity"] = {"real_txm": st(txm_p), "translated": st(fake_p),
                        "sem_source": st(sem_p)}
    for k, v in res["intensity"].items():
        print(f"   {k:12s} {v}")

    print("4. crack contrast retention ...")
    res["contrast"] = contrast_report(G, dev, rng)

    C.OUT.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"-> {args.out}")


def contrast_report(G, dev, rng, max_images=8):
    """Translate whole frames that have hand-drawn masks, and compare the local
    contrast of each marked crack region before and after."""
    from PIL import Image
    from translate import translate
    man = json.loads((C.CACHE / "manifest.json").read_text())
    rows = []
    done = 0
    for e in man["sem"]:
        m = C.SEM_MASK_DIR / f"{e['stem']}_correction_mask.png"
        if not m.exists() or done >= max_images:
            continue
        img = np.load(C.CACHE / "sem" / f"{e['stem']}.npy").astype(np.float32) / 255.0
        with Image.open(m) as im:
            mk = np.array(im)
        if mk.shape != img.shape:
            continue
        crack = mk == 1                      # 1 = hand-added crack
        if crack.sum() < 2000:
            continue
        out = translate(G, img, dev)
        before = crack_contrast(img, crack, rng=rng)
        after = crack_contrast(out, crack, rng=rng)
        if not before or not after:
            continue
        b = np.array([x[0] for x in before])
        a = np.array([x[0] for x in after])
        k = min(len(b), len(a))
        rows.append({
            "image": e["stem"],
            "n_regions": k,
            "sem_mean_contrast": round(float(b[:k].mean()), 4),
            "translated_mean_contrast": round(float(a[:k].mean()), 4),
            "frac_still_darker": round(float((a[:k] < 0).mean()), 4),
            "pearson": round(float(np.corrcoef(b[:k], a[:k])[0, 1]), 4) if k > 2 else None,
        })
        print(f"   {e['stem'][:44]:44s} n={k:4d} "
              f"SEM {b[:k].mean():+.4f} -> TXM {a[:k].mean():+.4f} "
              f"darker {(a[:k] < 0).mean():.0%}")
        done += 1
    if rows:
        print(f"   pooled: {np.mean([r['frac_still_darker'] for r in rows]):.0%} "
              f"of marked cracks remain darker than their surroundings")
    return rows


if __name__ == "__main__":
    main()
