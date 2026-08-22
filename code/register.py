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


def refine(a01, b01, max_features=4000, min_samples=4, residual_threshold=3.0,
           transform="similarity", verbose=True):
    """Fit a transform mapping a01 onto b01 from matched keypoints, with RANSAC.

    The coarse template search returns scale + translation only. Real pairs need
    rotation and (for an affine) shear too, and -- more importantly -- they need a
    RESIDUAL, because the residual is what decides the loss. TXM2SEM can use a
    pixel L1 term because FIB-SEM milling registers its pairs for free; a surface
    SEM against a projection radiograph has no such guarantee, and an L1 applied
    over a 20 px misalignment teaches blur rather than translation.

    MUST be given a coarsely localised window, not two whole frames. Measured on a
    known-positive pair (260618_b2_343_75 against its own LARGE field of view):
    matching against the full LARGE frame yields 92 matches, 8 inliers and a
    nonsense fit (scale 0.835, rotation +21 deg), because ORB has far too many
    candidate destinations. Matching against the window the coarse template search
    already localised yields 116 matches, 11 inliers, scale 1.0020, rotation
    -0.38 deg and a median residual of 1.07 px. Same images, same code; the only
    difference is search extent. So the pipeline is coarse-then-fine, and
    register_and_refine() enforces that ordering.

    Returns a dict with the fitted scale/rotation/translation, the inlier count,
    and the median inlier residual in pixels. Reads as a refusal when the inlier
    count is small: RANSAC will always fit something to enough noise.
    """
    from skimage.feature import ORB, match_descriptors
    from skimage.measure import ransac
    from skimage.transform import AffineTransform, SimilarityTransform
    from skimage import exposure

    def prep(z):
        # Equalise before detection: these are flat-fielded images with an IQR
        # around 0.02, and ORB finds almost nothing on raw contrast that low.
        return exposure.equalize_adapthist(np.asarray(z, np.float64), clip_limit=0.02)

    A, B = prep(a01), prep(b01)
    orb = ORB(n_keypoints=max_features, fast_threshold=0.02)
    try:
        orb.detect_and_extract(A)
        ka, da = orb.keypoints, orb.descriptors
        orb.detect_and_extract(B)
        kb, db = orb.keypoints, orb.descriptors
    except Exception as e:
        return {"ok": False, "reason": f"feature extraction failed: {e}"}
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return {"ok": False, "reason": "too few keypoints"}

    matches = match_descriptors(da, db, cross_check=True, max_ratio=0.9)
    if len(matches) < min_samples * 3:
        return {"ok": False, "reason": f"only {len(matches)} matches"}

    src = ka[matches[:, 0]][:, ::-1]      # (x, y)
    dst = kb[matches[:, 1]][:, ::-1]
    Model = SimilarityTransform if transform == "similarity" else AffineTransform
    try:
        model, inliers = ransac((src, dst), Model, min_samples=min_samples,
                                residual_threshold=residual_threshold,
                                max_trials=3000, rng=0)
    except Exception as e:
        return {"ok": False, "reason": f"ransac failed: {e}"}
    if model is None or inliers is None or inliers.sum() < min_samples * 2:
        return {"ok": False, "reason": "no consensus set"}

    pred = model(src[inliers])
    resid = np.hypot(*(pred - dst[inliers]).T)
    out = {
        "ok": True,
        "n_matches": int(len(matches)),
        "n_inliers": int(inliers.sum()),
        "inlier_frac": round(float(inliers.mean()), 3),
        "median_residual_px": round(float(np.median(resid)), 2),
        "p90_residual_px": round(float(np.percentile(resid, 90)), 2),
        "translation_px": [round(float(v), 2) for v in model.translation],
    }
    if hasattr(model, "scale") and np.isscalar(model.scale):
        out["scale"] = round(float(model.scale), 4)
        out["rotation_deg"] = round(float(np.rad2deg(model.rotation)), 3)
    if verbose:
        print(f"    matches {out['n_matches']}  inliers {out['n_inliers']} "
              f"({out['inlier_frac']:.0%})  median residual "
              f"{out['median_residual_px']} px  p90 {out['p90_residual_px']} px")
        if "scale" in out:
            print(f"    scale {out['scale']}  rotation {out['rotation_deg']} deg  "
                  f"translation {out['translation_px']}")
    return out


def register_and_refine(a01, b01, ratios, angles=(0.0,), verbose=True):
    """Coarse template search, then RANSAC refinement inside the located window.

    Returns (coarse, refined). Either can be a refusal; the refinement is only
    attempted when the coarse match clears the independence floor, because
    refining inside a window that is not the right window just fits noise.
    """
    from skimage.transform import rescale
    rows = []
    for r in ratios:
        for ang in angles:
            m = best_match(a01, b01, r, ang)
            if m is not None:
                rows.append({"ratio": float(r), "angle": float(ang),
                             "nmi": round(m[0], 4), "at": m[1],
                             "template": list(m[2]), "ncc": round(m[3], 4)})
    if not rows:
        return None, {"ok": False, "reason": "no candidate template fits"}
    rows.sort(key=lambda d: -d["nmi"])
    coarse = rows[0]
    coarse["margin_over_next"] = (round(coarse["nmi"] - rows[1]["nmi"], 4)
                                  if len(rows) > 1 else None)
    if verbose:
        print(f"  coarse: ratio {coarse['ratio']} angle {coarse['angle']} "
              f"NMI {coarse['nmi']} NCC {coarse['ncc']} at {coarse['at']}")
    if coarse["nmi"] < NMI_FLOOR:
        return coarse, {"ok": False,
                        "reason": f"coarse NMI {coarse['nmi']} is at the "
                                  f"independence floor; not refining noise"}
    y, x = coarse["at"]
    h, w = coarse["template"]
    win = b01[y:y + h, x:x + w]
    a_scaled = rescale(a01, 1.0 / coarse["ratio"], anti_aliasing=True,
                       preserve_range=True)[:h, :w]
    if win.shape != a_scaled.shape:
        hh = min(win.shape[0], a_scaled.shape[0])
        ww = min(win.shape[1], a_scaled.shape[1])
        win, a_scaled = win[:hh, :ww], a_scaled[:hh, :ww]
    return coarse, refine(a_scaled, win, verbose=verbose)


def loss_advice(refined, coarse=None):
    """Turn a residual into the decision it exists to inform.

    Inlier COUNT is a poor gate on its own. Measured on the known-positive pair:
    ORB gave 11 inliers with scale 1.0020 and rotation -0.385 deg (coherent, and
    matching the coarse search's independent estimate of ratio 1.00), while SIFT on
    the same pair gave 9 inliers with scale 0.9483 and rotation +1.371 deg
    (incoherent). Similar counts, opposite trustworthiness.

    So the gate is AGREEMENT between two independent estimates -- the coarse
    template search's scale and the refinement's scale -- plus the residual. Two
    methods that disagree about scale have not registered anything, however many
    inliers either reports.
    """
    if not refined.get("ok"):
        return "Registration failed. No paired loss is available; stay unpaired."
    r = refined["median_residual_px"]
    n = refined["n_inliers"]
    notes = []

    if coarse is not None and "scale" in refined:
        implied = 1.0 / coarse["ratio"]
        got = refined["scale"] / coarse["ratio"]
        disagree = abs(got - implied) / max(implied, 1e-9)
        if disagree > 0.05:
            return (f"Coarse search says pixel ratio {coarse['ratio']:.3f}; the "
                    f"refinement implies {coarse['ratio'] / max(refined['scale'], 1e-9):.3f} "
                    f"({disagree:.0%} apart). Two independent estimates disagreeing "
                    f"means neither has registered anything. Do not use a paired "
                    f"loss.")
        notes.append(f"coarse and refined scale agree to {disagree:.1%}")

    if n < 20:
        notes.append(f"only {n} inliers, so treat the transform as provisional")

    tail = ("  (" + "; ".join(notes) + ")") if notes else ""
    if r <= 2.0:
        return (f"Median residual {r} px. Sub-pixel-ish: a pixel L1 / pix2pix term "
                f"is appropriate, as in TXM2SEM.{tail}")
    if r <= 8.0:
        return (f"Median residual {r} px. Too coarse for pixel L1, which would "
                f"teach blur at that offset. Use a patch-level or contextual loss, "
                f"or downsample by ~{int(np.ceil(r / 2))}x so the residual is ~2 px "
                f"at the working resolution.{tail}")
    return (f"Median residual {r} px. Far too coarse for any pixel-aligned loss. "
            f"Use distribution-level supervision only, or re-register.{tail}")


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
    sem = np.load(C.CACHE / "sem" / f"{args.sem}.npy").astype(np.float32) / 255.0
    txm = np.load(C.CACHE / "txm" / f"{args.txm}.npy").astype(np.float32) / 255.0
    coarse, refined = register_and_refine(sem, txm, ratios, angles)
    if coarse is None:
        print("  no candidate ratio produced a usable template")
        return
    print(f"\n  {loss_advice(refined, coarse)}")
    if args.overlay and coarse.get("nmi", 0) >= NMI_FLOOR:
        write_overlay(args.sem, args.txm, coarse, args.overlay)
        print(f"  overlay -> {args.overlay}")
    elif args.overlay:
        print("  (no overlay written: coarse match is at the independence floor)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"coarse": coarse, "refined": refined,
                       "advice": loss_advice(refined, coarse)}, f, indent=1)
        print(f"  -> {args.out}")


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
