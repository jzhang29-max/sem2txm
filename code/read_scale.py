"""Recover the SEM pixel size that the TIFF tags lost.

The shipped micrographs carry no FEI/ZEISS pixel-size tag -- the SEM tool's own
provenance records `"calibrated": false, "um_per_px": null`. But the instrument
burned the numbers into the bottom of the frame: date, kV, current, WD, HFW,
detector, and a drawn scale bar with its length in microns.

So the calibration is not missing, it is just not machine-readable. This script
crops that panel out at full resolution so the two numbers can be read off it,
and measures the drawn bar in pixels so that only its LABEL has to be typed:

    um_per_px = (bar label in um) / (measured bar length in px)

Cross-check against the printed HFW:  um_per_px = HFW / image_width_px.
Two independent readings of the same quantity; if they disagree, do not use
either -- which is the convention semcrack.py already follows.
"""
import argparse
import sys

import numpy as np
import tifffile
from PIL import Image

import config as C

sys.path.insert(0, str(C.SEM_REPO / "code"))
from detect_cracks import _detect_databar_top  # noqa: E402


def from_metadata(path):
    """Exact SEM pixel size from the FEI/Thermo tag, when the file still has it.

    The repo copies were rewritten by tifffile and lost this, which is why the
    burned-in bar had to be measured at all. Originals straight off the Apreo do
    carry it, and it is exact -- no 100 um assumption. Returns (um_per_px, hfw_um)
    or (None, None).

    Cross-checked against the bar on two frames at different magnifications:
    metadata 0.103766 vs bar 0.103842 um/px (0.07% apart), and metadata 0.042155 vs
    bar 0.042159 (0.01%). So the bar method is sound where metadata is absent, and
    the bars really are 100 um.
    """
    import tifffile
    try:
        with tifffile.TiffFile(str(path)) as t:
            tags = t.pages[0].tags
            if "FEI_HELIOS" not in tags:
                return None, None
            md = tags["FEI_HELIOS"].value
            um = float(md["Scan"]["PixelWidth"]) * 1e6
            hfw = float(md.get("EBeam", {}).get("HFW", 0.0)) * 1e6 or None
            return um, hfw
    except Exception:
        return None, None


def databar(path):
    raw = tifffile.imread(str(path))
    if raw.ndim == 3:
        raw = raw.mean(axis=2)
    lo, hi = np.percentile(raw, [1.0, 99.5])
    img8 = np.clip((raw - lo) / max(hi - lo, 1e-8) * 255, 0, 255).astype(np.uint8)
    top = _detect_databar_top(img8)
    return img8[top:], top, raw.shape


def measure_bar(bar8, min_run=200):
    """Length of the drawn scale bar, in pixels.

    The bar is drawn with its own label inside it -- |----100 um----| -- so the
    longest single run of light pixels is only HALF the bar, which is how this
    first read 0.10 um/px for a frame whose panel says HFW 259 um. What is wanted
    is the SPAN from the start of the first long run to the end of the last one on
    the row that carries them.

    Verified on 260622_316_H_b2_front_CBS_01: span 2372 px, and 100 um / 2372 px
    = 0.0422 um/px, which puts HFW at 259 um -- the number printed on that same
    panel. Two independent readings of one quantity agreeing is the check; a row
    that yields only one long run is a panel divider, not a scale bar, and is
    rejected rather than measured.

    Returns (span_px, row) or (None, None).
    """
    if bar8.size == 0 or bar8.shape[0] < 3:
        return None, None          # no panel detected on this frame
    thr = max(200, int(np.percentile(bar8, 99)) - 20)
    best = (0, None)
    for r in range(bar8.shape[0]):
        row = bar8[r] >= thr
        if not row.any():
            continue
        d = np.diff(np.concatenate([[0], row.view(np.int8), [0]]))
        starts = np.flatnonzero(d == 1)
        lens = np.flatnonzero(d == -1) - starts
        long = [(int(a), int(b)) for a, b in zip(starts, lens) if b > min_run]
        if len(long) < 2:
            continue                     # one long run == a divider rule
        span = long[-1][0] + long[-1][1] - long[0][0]
        if span > best[0]:
            best = (int(span), r)
    return best if best[1] is not None else (None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="*", help="SEM stems (default: a few representative)")
    ap.add_argument("--bar-um", type=float, default=None,
                    help="the microns printed on the scale bar, to convert")
    ap.add_argument("--out", default=str(C.FIGURES / "databars"))
    ap.add_argument("--json", default=str(C.OUT / "sem_scale.json"),
                    help="write the per-frame table here")
    args = ap.parse_args()

    files = C.sem_files()
    if args.stems:
        files = [f for f in files if f.stem in args.stems]
    else:
        files = files[:6]
    from pathlib import Path
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"{'frame':46s} {'image':>13s} {'bar_px':>7s}")
    groups, per_frame = {}, []
    for f in files:
        bar8, top, shape = databar(f)
        L, row = measure_bar(bar8)
        if bar8.size:
            Image.fromarray(bar8).save(outdir / f"{f.stem}_databar.png")
        print(f"{f.stem[:46]:46s} {str(shape):>13s} "
              f"{(str(L) if L else 'no panel'):>8s}")
        per_frame.append({"frame": f.stem, "shape": list(shape), "bar_px": L,
                          "um_per_px": round(args.bar_um / L, 6) if (L and args.bar_um) else None})
        if L:
            groups.setdefault((L, shape[1]), []).append(f.stem)
        if L and args.bar_um:
            print(f"{'':46s} -> {args.bar_um / L:.5f} um/px "
                  f"(implies HFW {args.bar_um / L * shape[1]:.0f} um)")

    print(f"\n{len(groups)} distinct (bar_px, width) settings -- the bar's LABEL still has "
          f"to be read\nonce per setting, from the panels written to {outdir}:")
    for (L, w), stems in sorted(groups.items()):
        print(f"  bar {L:5d} px on {w} px wide : {len(stems):2d} frame(s), e.g. {stems[0][:46]}")
        if args.bar_um:
            print(f"      with --bar-um {args.bar_um:g}: {args.bar_um / L:.5f} um/px, "
                  f"HFW {args.bar_um / L * w:.0f} um")
    print("\nCross-check: um_per_px x image_width should equal the HFW printed on the panel.")

    if args.json:
        import json
        from pathlib import Path as _P
        _P(args.json).parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "assumption": ("every drawn scale bar is labelled 100 um; confirmed for "
                           "260622_316_H_b2_front_CBS_01, whose panel prints HFW 259 um "
                           "and whose 2372 px bar gives 100/2372*6144 = 259 um. NOT "
                           "verified for the other magnifications -- read their panels."),
            "bar_um_assumed": args.bar_um,
            "frames": per_frame,
            "settings": [
                {"bar_px": L, "width_px": w, "n_frames": len(st),
                 "um_per_px": round(args.bar_um / L, 6) if args.bar_um else None,
                 "implied_hfw_um": round(args.bar_um / L * w, 1) if args.bar_um else None,
                 "frames": st}
                for (L, w), st in sorted(groups.items())],
        }
        with open(args.json, "w") as fh:
            json.dump(rec, fh, indent=1)
        print(f"-> {args.json}")


if __name__ == "__main__":
    main()
