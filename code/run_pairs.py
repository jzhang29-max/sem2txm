"""Ingest registered-pair candidates and answer the questions that need them.

Reads pairs.json (see pairs.example.json) and, for each pair:

  1. preprocesses both frames the way training did, so nothing is compared across
     a different pipeline
  2. coarse-registers SEM into the TXM frame over scale and rotation, scored by
     mutual information rather than grey-value correlation -- the modalities
     measure different physical quantities, so their intensities are not related
     by any monotone function
  3. refuses to go further if the coarse match sits at the measured independence
     floor (NMI 1.01; two unrelated fields score 1.004, and a same-specimen
     SEM/TXM attempt on the existing data scored 1.0005-1.006)
  4. refines with RANSAC inside the located window and reports the RESIDUAL, which
     is what decides whether a pixel-aligned loss is usable at all
  5. measures predicted-vs-real fidelity against two baselines -- the raw SEM and a
     blurred SEM -- because the translator has earned nothing until it beats doing
     nothing
  6. writes a checkerboard overlay, because a number cannot tell you a registration
     is real and a checkerboard can

If txm_um_per_px is supplied it also reports the SEM:TXM pixel ratio, which is the
one quantity the whole project has been missing.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import config as C


def load_and_prep(path, kind):
    from predict import read_image, preprocess
    raw = read_image(path)
    img, box = preprocess(raw, crop_panel=(kind == "sem"))
    return img, raw.shape, box


def sem_um_per_px(path):
    """SEM scale: instrument metadata first, burned-in bar as fallback.

    Metadata is exact and carries no assumption; the bar assumes 100 um, which was
    verified against metadata to 0.07% on two frames at different magnifications.
    Returns (um_per_px, source, detail).
    """
    from read_scale import from_metadata, databar, measure_bar
    um, hfw = from_metadata(path)
    if um:
        return um, "metadata", f"HFW {hfw:.1f} um" if hfw else ""
    try:
        bar, top, shape = databar(path)
        L, _ = measure_bar(bar)
        if L:
            return 100.0 / L, "scale bar", f"bar {L} px of {shape[1]}"
    except Exception:
        pass
    return None, None, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=str(C.ROOT / "pairs.json"))
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--ratios", default="1,2,3,4,6,8,10,12,16")
    ap.add_argument("--angles", default="-6,-3,0,3,6")
    ap.add_argument("--out", default=str(C.OUT / "pairs"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    pf = Path(args.pairs)
    if not pf.exists():
        raise SystemExit(f"no {pf}. Copy pairs.example.json to pairs.json and fill "
                         f"it in.")
    spec = json.loads(pf.read_text())
    base = pf.parent
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    from register import register_and_refine, loss_advice, write_overlay, NMI_FLOOR
    from eval_paired import metrics, apply_registration
    from predict import pick_checkpoint
    from train import device_of
    from translate import load_generator, translate
    from scipy import ndimage as ndi

    ckpt = pick_checkpoint(args.ckpt)
    dev = device_of(args.device)
    G, ck = load_generator(ckpt, dev)
    print(f"model {ckpt.name} (iter {ck['iter']}) on {dev}\n")

    txm_scale = spec.get("txm_um_per_px")
    by_frame = spec.get("txm_um_per_px_by_frame") or {}
    ratios = [float(x) for x in args.ratios.split(",")]
    angles = [float(x) for x in args.angles.split(",")]
    results = []

    for pr in spec.get("pairs", []):
        name = pr.get("name") or "pair"
        sp = Path(pr["sem"]);  sp = sp if sp.is_absolute() else base / sp
        tp = Path(pr["txm"]);  tp = tp if tp.is_absolute() else base / tp
        print(f"=== {name}: {sp.name}  ->  {tp.name}")
        if not sp.exists() or not tp.exists():
            print(f"    missing file; skipped")
            continue
        sem, sem_raw_shape, _ = load_and_prep(sp, "sem")
        txm, _, _ = load_and_prep(tp, "txm")
        print(f"    SEM {sem.shape}   TXM {txm.shape}")

        # scale, if it can be established
        s_um, src, detail = sem_um_per_px(sp)
        t_um = by_frame.get(tp.name, txm_scale)
        if s_um:
            print(f"    SEM scale ({src}): {s_um:.6f} um/px  {detail}")
        if s_um and t_um:
            print(f"    TXM scale given: {t_um} um/px  ->  expected pixel ratio "
                  f"{t_um / s_um:.2f}  (SEM must be downsampled by this)")
            ratios_use = sorted({round(t_um / s_um, 2), *ratios})
        else:
            ratios_use = ratios
            if not t_um:
                print(f"    no TXM um/px supplied -- searching scale blind")

        coarse, refined = register_and_refine(sem, txm, ratios_use, angles)
        if coarse is None:
            print("    no candidate template fits\n")
            continue
        advice = loss_advice(refined, coarse)
        print(f"    {advice}")

        rec = {"name": name, "sem": str(sp), "txm": str(tp), "coarse": coarse,
               "refined": refined, "advice": advice,
               "sem_um_per_px": s_um, "sem_scale_source": src, "txm_um_per_px": t_um}
        # With the SEM scale known exactly, a registration DERIVES the TXM scale:
        # the fitted ratio is how many SEM pixels make one TXM pixel.
        if s_um and refined.get("ok"):
            r_eff = coarse["ratio"] / max(refined.get("scale", 1.0), 1e-9)
            rec["txm_um_per_px_derived"] = round(r_eff * s_um, 6)
            print(f"    ==> TXM scale DERIVED from this registration: "
                  f"{r_eff * s_um:.4f} um/px  (ratio {r_eff:.3f} x SEM)")

        if coarse["nmi"] >= NMI_FLOOR:
            ov = outdir / f"{name}_overlay.png"
            try:
                write_overlay_direct(sem, txm, coarse, ov)
                print(f"    overlay -> {ov.name}   <-- LOOK AT THIS before believing "
                      f"the numbers")
            except Exception as e:
                print(f"    overlay failed: {e}")
            # fidelity of the prediction against the real TXM it landed on
            pred = translate(G, sem, dev, 512, 256, 8, offsets=2)
            y, x = coarse["at"]
            pt = apply_registration(pred, coarse)
            st = apply_registration(sem, coarse)
            h, w = pt.shape
            truth = txm[y:y + h, x:x + w]
            if truth.shape == pt.shape:
                rows = {"predicted TXM": metrics(pt, truth),
                        "input SEM (do nothing)": metrics(st, truth),
                        "blurred SEM": metrics(ndi.gaussian_filter(st, 2.0), truth)}
                print(f"    {'':26s} {'SSIM':>8s} {'pearson':>9s} {'NMI':>7s}")
                for k, v in rows.items():
                    print(f"    {k:26s} {v['ssim']:8.4f} {v['pearson']:9.4f} "
                          f"{v['nmi']:7.4f}")
                best = max(rows["input SEM (do nothing)"]["nmi"],
                           rows["blurred SEM"]["nmi"])
                got = rows["predicted TXM"]["nmi"]
                print(f"    -> translation {'BEATS' if got > best else 'does NOT beat'}"
                      f" doing nothing on NMI ({got:.4f} vs {best:.4f})")
                rec["fidelity"] = rows
        print()
        results.append(rec)

    with open(outdir / "pairs_report.json", "w") as f:
        json.dump({"model_iter": ck["iter"], "pairs": results}, f, indent=1)
    print(f"-> {outdir / 'pairs_report.json'}")
    ok = [r for r in results if r["refined"].get("ok")]
    print(f"\n{len(ok)} of {len(results)} pairs registered above the independence "
          f"floor.")
    if ok:
        res = [r["refined"]["median_residual_px"] for r in ok]
        print(f"residuals: {res}  -- these decide the loss, see the advice above.")


def write_overlay_direct(sem, txm, coarse, path):
    """Checkerboard, from arrays already in memory."""
    from PIL import Image
    from skimage.transform import rescale, rotate as skrotate
    t = rescale(sem, 1.0 / coarse["ratio"], anti_aliasing=True, preserve_range=True)
    if coarse.get("angle"):
        t = skrotate(t, coarse["angle"], resize=False, mode="reflect",
                     preserve_range=True)
        cut = int(np.ceil(abs(np.sin(np.deg2rad(coarse["angle"]))) * max(t.shape))) + 2
        if 2 * cut < min(t.shape):
            t = t[cut:-cut, cut:-cut]
    y, x = coarse["at"]
    h, w = coarse["template"]
    win = txm[y:y + h, x:x + w]
    t = t[:win.shape[0], :win.shape[1]]
    if win.shape != t.shape:
        raise ValueError("window/template shape mismatch")
    def n(a):
        lo, hi = np.percentile(a, [1, 99])
        return np.clip((a - lo) / max(hi - lo, 1e-8), 0, 1)
    a, b = n(t), n(win)
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    chk = (((yy // 64) + (xx // 64)) % 2).astype(bool)
    sheet = np.concatenate([a, b, np.where(chk, a, b)], axis=1)
    Image.fromarray((sheet * 255).astype(np.uint8)).save(path)


if __name__ == "__main__":
    main()
