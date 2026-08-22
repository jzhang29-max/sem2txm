"""Recover TXM um/px from the stage positions encoded in the filenames.

The TXM pixel size is the most consequential unknown in this repo: training runs at
1:1 pixels against SEM, so if the ratio is far from 1 the generator has been
matching mismatched fields of view. The .xrm metadata did not survive conversion
and those mosaics carry no burned-in panel, so it looked unrecoverable.

But the filenames carry it. The 260618_b2 series runs 333_75, 335_31, 336_25,
337_19, 338_13, 339_06, 340_00, 340_94, 341_88, 342_81, 343_75 -- read as microns
those are 333.75 ... 343.75, stepping by a near-constant 0.9375 um. If that is a
STAGE TRANSLATION, consecutive frames are the same field shifted by a known
physical distance, and registering them measures that distance in pixels:

    um_per_px = delta_um / shift_px

This is the same trick as reading the SEM scale bar: find a known physical length
and count the pixels across it.

Two ways this can fail, and both are detectable rather than silent:

  - The positions may be a DEPTH or focus axis, not in-plane. Then consecutive
    frames show no consistent translation and the fitted shift is ~0 with no
    correlation against delta_um. Reported, not assumed.
  - Phase correlation returns a shift for any pair of images. So the fit across
    ALL consecutive pairs is what is trusted, not any single pair: a real stage
    translation gives shift proportional to delta_um through the origin, and noise
    does not.
"""
import argparse
import re

import numpy as np

import config as C


def series(prefix="260618_b2"):
    """Cached TXM stems whose name encodes a micron position, sorted by it."""
    out = []
    for p in sorted((C.CACHE / "txm").glob("*.npy")):
        if p.name.endswith(".valid.npy"):
            continue
        stem = p.stem
        if not stem.lower().startswith(prefix.lower()):
            continue
        m = re.match(rf"^{re.escape(prefix)}_(\d+)_(\d+)$", stem, re.I)
        if not m:
            continue
        out.append((float(f"{m.group(1)}.{m.group(2)}"), stem))
    return sorted(out)


def shift_between(a_stem, b_stem, upsample=20):
    """Sub-pixel translation between two same-modality frames."""
    from skimage.registration import phase_cross_correlation
    a = np.load(C.CACHE / "txm" / f"{a_stem}.npy").astype(np.float32) / 255.0
    b = np.load(C.CACHE / "txm" / f"{b_stem}.npy").astype(np.float32) / 255.0
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    # Mask the mosaic no-data region: zeros dominate a correlation.
    valid = (a > 0) & (b > 0)
    if valid.mean() < 0.5:
        return None
    a = np.where(valid, a, float(a[valid].mean()))
    b = np.where(valid, b, float(b[valid].mean()))
    shift, error, _ = phase_cross_correlation(a, b, upsample_factor=upsample,
                                              normalization=None)
    return np.asarray(shift, float), float(error), (h, w)


def selftest():
    """Shift a real frame by a known amount and check it is recovered."""
    from skimage.registration import phase_cross_correlation
    st = series()
    if not st:
        return
    a = np.load(C.CACHE / "txm" / f"{st[0][1]}.npy").astype(np.float32) / 255.0
    a = a[:1024, :1024]
    for truth in ((7, -3), (25, 11)):
        b = np.roll(a, truth, axis=(0, 1))
        got, _, _ = phase_cross_correlation(a, b, upsample_factor=20,
                                            normalization=None)
        print(f"  self-test: applied {truth}, recovered "
              f"{tuple(np.round(-got, 2))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="260618_b2")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("phase correlation self-test on a real frame:")
        selftest()
        print()

    st = series(args.prefix)
    print(f"{len(st)} frames in series '{args.prefix}' with a micron position:")
    for um, stem in st:
        print(f"   {um:8.2f} um   {stem}")
    if len(st) < 3:
        print("  too few to fit")
        return

    print(f"\nconsecutive-pair registration:")
    d_um, d_px, rows = [], [], []
    for (u0, s0), (u1, s1) in zip(st, st[1:]):
        r = shift_between(s0, s1)
        if r is None:
            print(f"   {u0:.2f} -> {u1:.2f}: too little valid overlap")
            continue
        shift, err, shape = r
        mag = float(np.hypot(*shift))
        du = u1 - u0
        d_um.append(du); d_px.append(mag)
        rows.append((u0, u1, du, shift, mag, err))
        print(f"   {u0:7.2f} -> {u1:7.2f}  d={du:5.2f} um   "
              f"shift=({shift[0]:+8.2f},{shift[1]:+8.2f}) px  |shift|={mag:7.2f}")

    if len(d_um) < 3:
        print("  not enough pairs to fit")
        return
    d_um = np.array(d_um); d_px = np.array(d_px)
    # Fit through the origin: a stage translation has no offset.
    slope = float((d_um * d_px).sum() / max((d_um ** 2).sum(), 1e-12))   # px per um
    resid = d_px - slope * d_um
    ss = float(1 - (resid ** 2).sum() / max(((d_px - d_px.mean()) ** 2).sum(), 1e-12))
    corr = float(np.corrcoef(d_um, d_px)[0, 1]) if len(d_um) > 2 else float("nan")

    print(f"\nfit through origin: {slope:.3f} px per um   R^2={ss:+.3f}   "
          f"corr(d_um, |shift|)={corr:+.3f}")
    print(f"median |shift| {np.median(d_px):.2f} px for a median step of "
          f"{np.median(d_um):.3f} um")

    if abs(corr) < 0.5 or slope <= 0.05:
        print("\nVERDICT: these positions are NOT an in-plane stage translation.")
        print("  Consecutive frames do not shift in proportion to the position")
        print("  difference, so the axis is depth/focus (or the label means")
        print("  something else) and no um/px follows from it.")
        if np.median(d_px) < 2.0:
            print("  Median shift is under 2 px: the frames are essentially")
            print("  co-located, consistent with a through-thickness series.")
    else:
        print(f"\nVERDICT: consistent with an in-plane translation.")
        print(f"  TXM pixel size = 1/{slope:.3f} = {1.0/slope:.4f} um/px")


if __name__ == "__main__":
    main()
