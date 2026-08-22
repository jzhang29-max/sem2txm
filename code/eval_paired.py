"""Fidelity of a predicted TXM against a real one, on registered pairs.

This is the measurement the repo currently cannot make. Everything in
eval_translation.py is a proxy: whether the output looks like TXM (a two-sample
test) and whether it kept the input's structure (contrast correlation). Neither
asks the only question that matters for "predict what the TXM looks like" --
is the prediction RIGHT.

With a registered pair it becomes answerable, and the design point is the
baselines. A similarity score against the truth means nothing on its own, because
the input already resembles the target somewhat: both are images of the same
cracked steel. So every metric is reported three ways --

    predicted TXM   vs real TXM      what the model achieves
    input SEM       vs real TXM      what doing NOTHING achieves
    blurred SEM     vs real TXM      what a one-line baseline achieves

-- and the translator has earned its place only if column 1 beats columns 2 and 3.
A model that scores SSIM 0.6 where passthrough scores 0.62 is worse than useless,
and that is exactly the comparison a single headline number hides.

WHICH METRIC TO BELIEVE. These were checked against synthetic cases, and they
disagree in a way that matters here. A contrast-INVERTED prediction scores SSIM
-0.97 and Pearson -1.0, but NMI 2.0. Inversion is not necessarily an error across
these two modalities: a void attenuates less, so a feature that is dark in SEM
(a pit, in backscatter) can legitimately be bright in transmission. SSIM and PSNR
assume the prediction should match sign-for-sign and will condemn a physically
correct inversion; NMI asks only whether one image determines the other. So read
NMI as the primary number and SSIM as a secondary one, and if they disagree
sharply, check the sign before concluding the model failed.

Usage once a registration exists (see register.py):

    python code/eval_paired.py --sem <stem> --txm <stem> \\
        --registration out/reg_pair1.json --ckpt runs/cut/final.pt
"""
import argparse
import json

import numpy as np

import config as C


def apply_registration(img, reg, target_shape=None):
    """Put a SEM-space image into TXM space using a registration record."""
    from skimage.transform import rescale, rotate as skrotate
    t = rescale(np.asarray(img, np.float32), 1.0 / reg["ratio"],
                anti_aliasing=True, preserve_range=True)
    if reg.get("angle"):
        t = skrotate(t, reg["angle"], resize=False, mode="reflect",
                     preserve_range=True)
        cut = int(np.ceil(abs(np.sin(np.deg2rad(reg["angle"]))) * max(t.shape))) + 2
        if 2 * cut < min(t.shape):
            t = t[cut:-cut, cut:-cut]
    if target_shape is not None:
        t = t[:target_shape[0], :target_shape[1]]
    return t


def metrics(pred, truth):
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    from register import nmi
    pred = np.asarray(pred, np.float64)
    truth = np.asarray(truth, np.float64)
    # Match first and second moments before comparing: an overall brightness or
    # contrast offset is a display convention, not a prediction error, and SSIM
    # and PSNR both punish it heavily.
    p = (pred - pred.mean()) / max(pred.std(), 1e-8)
    t = (truth - truth.mean()) / max(truth.std(), 1e-8)
    p01 = np.clip(p * 0.2 + 0.5, 0, 1)
    t01 = np.clip(t * 0.2 + 0.5, 0, 1)
    return {
        "ssim": round(float(structural_similarity(t01, p01, data_range=1.0)), 4),
        "psnr": round(float(peak_signal_noise_ratio(t01, p01, data_range=1.0)), 2),
        "pearson": round(float(np.corrcoef(p.ravel(), t.ravel())[0, 1]), 4),
        "nmi": round(float(nmi(p01, t01)), 4),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sem", required=True)
    ap.add_argument("--txm", required=True)
    ap.add_argument("--registration", required=True,
                    help="JSON from register.py (ratio, angle, at)")
    ap.add_argument("--ckpt", default=str(C.ROOT / "runs" / "cut" / "final.pt"))
    ap.add_argument("--blur-sigma", type=float, default=2.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from scipy import ndimage as ndi
    from train import device_of
    from translate import load_generator, translate

    reg = json.loads(open(args.registration).read())
    if reg.get("nmi", 0) < 1.01:
        print("WARNING: this registration scores at the independence floor. "
              "Every number below is then meaningless -- fix the registration first.")

    sem = np.load(C.CACHE / "sem" / f"{args.sem}.npy").astype(np.float32) / 255.0
    txm = np.load(C.CACHE / "txm" / f"{args.txm}.npy").astype(np.float32) / 255.0

    dev = device_of(args.device)
    G, ck = load_generator(args.ckpt, dev)
    pred_full = translate(G, sem, dev)

    # Move both the prediction and the raw input into TXM space, then crop the
    # window the registration landed on.
    y, x = reg["at"]
    pred_t = apply_registration(pred_full, reg)
    sem_t = apply_registration(sem, reg)
    h, w = pred_t.shape
    truth = txm[y:y + h, x:x + w]
    if truth.shape != pred_t.shape:
        h, w = truth.shape
        pred_t, sem_t = pred_t[:h, :w], sem_t[:h, :w]

    blur_t = ndi.gaussian_filter(sem_t, args.blur_sigma)

    rows = {
        "predicted TXM": metrics(pred_t, truth),
        "input SEM (do nothing)": metrics(sem_t, truth),
        f"blurred SEM (sigma {args.blur_sigma:g})": metrics(blur_t, truth),
    }
    print(f"\ncheckpoint iter {ck['iter']}   overlap {h}x{w} px   "
          f"registration NMI {reg.get('nmi')}\n")
    print(f"{'':30s} {'SSIM':>8s} {'PSNR':>8s} {'pearson':>9s} {'NMI':>7s}")
    for k, v in rows.items():
        print(f"{k:30s} {v['ssim']:8.4f} {v['psnr']:8.2f} "
              f"{v['pearson']:9.4f} {v['nmi']:7.4f}")

    base = max(rows["input SEM (do nothing)"]["ssim"],
               rows[f"blurred SEM (sigma {args.blur_sigma:g})"]["ssim"])
    got = rows["predicted TXM"]["ssim"]
    print()
    if got > base:
        print(f"  The translation beats doing nothing on SSIM by {got - base:+.4f}.")
    else:
        print(f"  The translation does NOT beat doing nothing on SSIM "
              f"({got:+.4f} vs {base:+.4f}). On this pair it is not earning its "
              f"place.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"ckpt_iter": ck["iter"], "registration": reg,
                       "overlap": [h, w], "metrics": rows}, f, indent=1)
        print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
