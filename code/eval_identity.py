"""How much does the generator distort an image that is ALREADY TXM?

This is the one paired fidelity test available without cross-modality pairs, and
it is a genuine held-out one. The five reference-specimen mosaics are excluded from
the patch banks in prep.py, so the generator has never seen them; feeding one in
and comparing G(x) against x asks a question with a known answer.

It is a NECESSARY condition, not a sufficient one. A translator that mangles input
already in the target domain cannot be trusted to land a SEM frame there either --
so a bad score here condemns the model. A good score does not vindicate it, because
leaving TXM untouched is exactly what the identity-NCE term is trained to do, and
a generator could satisfy it while still doing something useless to SEM input.

Read it as a floor on distortion, and note it is measured on frames that were held
out, unlike the identity loss curve in training which is measured on frames that
were not.
"""
import argparse
import json

import numpy as np

import config as C
from eval_paired import metrics


def calibration(stem):
    """Known perturbations of a real frame, so the measured score can be read.

    An SSIM of 0.59 is not interpretable on its own. Applying perturbations of
    known severity to the same data gives a ladder to place it on, and the answer
    is not the one SSIM alone suggests: measured G(TXM) sits near blur sigma 1 on
    SSIM but near blur sigma 4 on both correlation and mutual information, which are
    the metrics that track information loss rather than local structure.
    """
    from scipy import ndimage as ndi
    x = np.load(C.CACHE / "txm" / f"{stem}.npy").astype(np.float32) / 255.0
    rng = np.random.default_rng(0)
    cases = [("gaussian blur sigma 1", ndi.gaussian_filter(x, 1.0)),
             ("gaussian blur sigma 2", ndi.gaussian_filter(x, 2.0)),
             ("gaussian blur sigma 4", ndi.gaussian_filter(x, 4.0)),
             ("gaussian blur sigma 8", ndi.gaussian_filter(x, 8.0)),
             ("additive noise sd 0.05", np.clip(x + 0.05 * rng.standard_normal(x.shape), 0, 1)),
             ("additive noise sd 0.10", np.clip(x + 0.10 * rng.standard_normal(x.shape), 0, 1)),
             ("shifted 2 px", np.roll(x, (2, 2), axis=(0, 1)))]
    print(f"\n  calibration ladder on {stem[:34]} -- known perturbations:")
    print(f"  {'perturbation':26s} {'SSIM':>7s} {'pearson':>9s} {'NMI':>7s}")
    for lab, y in cases:
        m = metrics(y, x)
        print(f"  {lab:26s} {m['ssim']:7.4f} {m['pearson']:9.4f} {m['nmi']:7.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(C.ROOT / "runs" / "cut" / "final.pt"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-px", type=int, default=6_000_000,
                    help="centre-crop frames larger than this, to bound runtime")
    ap.add_argument("--out", default=str(C.OUT / "eval_identity.json"))
    args = ap.parse_args()

    from train import device_of
    from translate import load_generator, translate

    man = json.loads((C.CACHE / "manifest.json").read_text())
    held = [e["stem"] for e in man["txm"] if e.get("reference")]
    dev = device_of(args.device)
    G, ck = load_generator(args.ckpt, dev)
    print(f"generator iter {ck['iter']} on {dev}")
    print(f"{len(held)} held-out reference mosaics (never in the training banks)\n")

    rows = {}
    for stem in held:
        x = np.load(C.CACHE / "txm" / f"{stem}.npy").astype(np.float32) / 255.0
        if x.size > args.max_px:
            s = int(np.sqrt(args.max_px / x.size) * min(x.shape))
            cy, cx = x.shape[0] // 2, x.shape[1] // 2
            x = x[max(0, cy - s // 2):cy + s // 2, max(0, cx - s // 2):cx + s // 2]
        y = translate(G, x, dev)
        m = metrics(y, x)
        rows[stem] = {"shape": list(x.shape), **m}
        print(f"  {stem[:38]:38s} {str(x.shape):>14s}  SSIM {m['ssim']:.4f}  "
              f"PSNR {m['psnr']:6.2f}  r {m['pearson']:+.4f}  NMI {m['nmi']:.4f}")

    if rows:
        calibration(list(rows)[0])
        ss = np.array([v["ssim"] for v in rows.values()])
        pr = np.array([v["pearson"] for v in rows.values()])
        nm = np.array([v["nmi"] for v in rows.values()])
        print(f"\n  mean over {len(rows)} held-out frames: SSIM {ss.mean():.4f} "
              f"+-{ss.std():.4f}   pearson {pr.mean():+.4f}   NMI {nm.mean():.4f}")
        print("\n  For scale: NMI 1.00 is independence, 2.00 is a deterministic")
        print("  relation. Pearson near +1 with high SSIM means the generator is")
        print("  close to a no-op on in-domain input, which is the desired")
        print("  behaviour of the identity branch.")
        C.OUT.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"ckpt_iter": ck["iter"], "frames": rows,
                       "mean_ssim": round(float(ss.mean()), 4),
                       "mean_pearson": round(float(pr.mean()), 4),
                       "mean_nmi": round(float(nm.mean()), 4)}, f, indent=1)
        print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
