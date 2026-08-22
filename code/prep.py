"""Stage 1: put both modalities into one comparable domain, then cut patch banks.

Why any preprocessing at all: a raw TXM mosaic is dominated by the beam/thickness
envelope, not by specimen structure. At 512 px a raw TXM crop is mostly a smooth
gradient (see figures/contact_txm_raw.png), so a translator trained raw->raw would
spend its capacity inventing an envelope and almost none on structure.

So both domains are put through the SAME operator family the TXM app already uses
for its human-facing view -- destitch (TXM only; SEM has no tile grid) then
pseudo-flat-field. Those modules are imported from the TXM repo rather than
reimplemented, so this tool's target domain is exactly the domain that repo's
reviewers mark cracks on. Both steps preserve geometry, so a SEM correction mask
still registers pixel-for-pixel after the transform -- which is what makes label
transfer possible at all.

The four dense-ground-truth TXM frames and their specimen siblings are excluded
here, not later: they are the test set for the label-transfer experiment, and a
translator that had seen their appearance would flatter it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

import config as C

sys.path.insert(0, str(C.TXM_REPO / "code"))
import destitch          # noqa: E402
import flatfield         # noqa: E402
from txm_features import robust_normalize  # noqa: E402

sys.path.insert(0, str(C.SEM_REPO / "code"))
from detect_cracks import find_field_of_view  # noqa: E402

# From TXM_Crack_Detection_Pipeline/app/core/pipeline.py: the dense-GT frames are
# that repo's test set, held out by SPECIMEN because two fields of view of one
# specimen leak into each other.
REFERENCE_SPECIMENS = ("333_75", "336_25", "338_13", "343_75")


def is_reference(path):
    s = C.txm_stem(path).lower()
    return any(t in s for t in REFERENCE_SPECIMENS)


def prep_txm_one(path, sigma_y=16.0, sigma_x=22.0):
    """destitch -> flat-field -> robust normalise. Returns (img01, valid)."""
    raw = tifffile.imread(str(path)).astype(np.float32)
    d, _ = destitch.destitch_image(raw)
    ff = flatfield.flatfield(np.asarray(d, np.float32), sigma_y=sigma_y, sigma_x=sigma_x)
    if isinstance(ff, tuple):
        ff = ff[0]
    ff = np.asarray(ff, np.float64)
    # flatfield() writes 0.0 into the no-data region outside the tile outline.
    valid = ff != 0.0
    if valid.sum() < 0.2 * valid.size:
        valid = np.ones_like(valid)
    lo, hi = np.percentile(ff[valid], [1.0, 99.0])
    img01 = np.clip((ff - lo) / max(hi - lo, 1e-8), 0, 1).astype(np.float32)
    img01[~valid] = 0.0
    return img01, valid


def sem_crop_box(raw):
    """The instrument burns an info panel into the bottom of these captures --
    date, kV, HFW, detector, scale bar. It is not specimen, and a translator
    trained on it learns to generate text. The SEM repo already has a detector
    for it that was validated across all 62 frames (including five whose panel
    is mid-grey rather than dark), so use that rather than a fresh guess.
    Returns (x0, y0, x1, y1)."""
    lo, hi = np.percentile(raw, [1.0, 99.5])
    img8 = np.clip((raw - lo) / max(hi - lo, 1e-8) * 255, 0, 255).astype(np.uint8)
    return find_field_of_view(img8)


def prep_sem_one(path, sigma=24.0, do_flatfield=True):
    """SEM needs no destitch (no tile grid), but gets the same flat-field so that
    the translation is about structure rather than about illumination envelope.
    The info panel is cropped off first."""
    raw = tifffile.imread(str(path)).astype(np.float32)
    if raw.ndim == 3:
        raw = raw.mean(axis=2)
    x0, y0, x1, y1 = sem_crop_box(raw)
    raw = raw[y0:y1, x0:x1]
    if do_flatfield:
        ff = flatfield.flatfield(raw, sigma_y=sigma, sigma_x=sigma)
        if isinstance(ff, tuple):
            ff = ff[0]
        ff = np.asarray(ff, np.float64)
    else:
        ff = raw.astype(np.float64)
    valid = np.ones(ff.shape, bool)
    img01 = robust_normalize(ff, 1.0, 99.0)
    return np.asarray(img01, np.float32), (x0, y0, x1, y1)


def build_cache(args):
    (C.CACHE / "txm").mkdir(parents=True, exist_ok=True)
    (C.CACHE / "sem").mkdir(parents=True, exist_ok=True)
    manifest = {"txm": [], "sem": []}

    txm = [f for f in C.txm_files() if not is_reference(f)]
    held = [f for f in C.txm_files() if is_reference(f)]
    print(f"TXM: {len(txm)} for training, {len(held)} held out as reference specimens")
    for f in held:
        print(f"   HELD OUT  {C.txm_stem(f)}")

    for i, f in enumerate(C.txm_files()):
        stem = C.txm_stem(f)
        out = C.CACHE / "txm" / f"{stem}.npy"
        if out.exists() and not args.force:
            a = np.load(out, mmap_mode="r")
        else:
            img01, valid = prep_txm_one(f)
            np.save(out, (img01 * 255).astype(np.uint8))
            np.save(C.CACHE / "txm" / f"{stem}.valid.npy", np.packbits(valid))
            a = img01
            print(f"  [{i+1}/{len(C.txm_files())}] {stem[:56]} {a.shape}", flush=True)
        manifest["txm"].append({"stem": stem, "group": C.txm_group(f),
                                "shape": list(a.shape), "reference": is_reference(f)})

    for i, f in enumerate(C.sem_files()):
        stem = f.stem
        out = C.CACHE / "sem" / f"{stem}.npy"
        if out.exists() and not args.force:
            a = np.load(out, mmap_mode="r")
        else:
            img01, box = prep_sem_one(f, do_flatfield=not args.no_sem_flatfield)
            np.save(out, (img01 * 255).astype(np.uint8))
            np.save(C.CACHE / "sem" / f"{stem}.box.npy", np.asarray(box))
            a = img01
            print(f"  [{i+1}/{len(C.sem_files())}] {stem[:56]} "
                  f"{tifffile.TiffFile(str(f)).pages[0].shape} -> {a.shape}", flush=True)
        box = np.load(C.CACHE / "sem" / f"{stem}.box.npy").tolist()
        manifest["sem"].append({"stem": stem, "group": C.sem_group(f),
                                "shape": list(a.shape), "crop": box})

    (C.CACHE / "manifest.json").write_text(json.dumps(manifest, indent=1))
    check_mask_alignment(manifest)
    print(f"cache -> {C.CACHE}")


def check_mask_alignment(manifest):
    """The hand-drawn correction masks were painted on the CROPPED frame, so a
    correct crop reproduces their shape exactly. Any mismatch means a transferred
    label would land on the wrong pixels, so it is checked rather than assumed."""
    from PIL import Image
    ok = bad = missing = 0
    for e in manifest["sem"]:
        m = C.SEM_MASK_DIR / f"{e['stem']}_correction_mask.png"
        if not m.exists():
            missing += 1
            continue
        with Image.open(m) as im:
            mshape = (im.size[1], im.size[0])
        if tuple(e["shape"]) == mshape:
            ok += 1
        else:
            bad += 1
            print(f"  MASK MISMATCH {e['stem']}: cropped {tuple(e['shape'])} "
                  f"vs mask {mshape}")
    print(f"mask alignment: {ok} match, {bad} mismatch, {missing} without a mask")


def cut_bank(domain, entries, per_image, patch, seed, exclude_reference=True):
    """Random patches into one uint8 array. Patches overlapping TXM no-data are
    rejected rather than zero-filled."""
    rng = np.random.default_rng(seed)
    keep = [e for e in entries if not (exclude_reference and e.get("reference"))]
    buf = np.zeros((len(keep) * per_image, patch, patch), np.uint8)
    src = []
    k = 0
    for e in keep:
        a = np.load(C.CACHE / domain / f"{e['stem']}.npy", mmap_mode="r")
        h, w = a.shape
        if h < patch or w < patch:
            continue
        got = 0
        for _ in range(per_image * 30):
            if got >= per_image:
                break
            y = int(rng.integers(0, h - patch))
            x = int(rng.integers(0, w - patch))
            p = np.asarray(a[y:y + patch, x:x + patch])
            if (p == 0).mean() > 0.02:      # mosaic no-data
                continue
            buf[k] = p
            src.append((e["stem"], e["group"], y, x))
            k += 1
            got += 1
    buf = buf[:k]
    np.save(C.CACHE / f"bank_{domain}.npy", buf)
    (C.CACHE / f"bank_{domain}_src.json").write_text(json.dumps(src))
    print(f"bank_{domain}: {k} patches of {patch}px from {len(keep)} images "
          f"({buf.nbytes/1e9:.2f} GB)")
    return buf


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="recompute cached images")
    ap.add_argument("--no-sem-flatfield", action="store_true",
                    help="leave the SEM illumination envelope in place")
    ap.add_argument("--per-image", type=int, default=300)
    ap.add_argument("--patch", type=int, default=C.PATCH)
    ap.add_argument("--bank-only", action="store_true")
    args = ap.parse_args()

    if not args.bank_only:
        build_cache(args)
    man = json.loads((C.CACHE / "manifest.json").read_text())
    cut_bank("sem", man["sem"], args.per_image, args.patch, 1)
    cut_bank("txm", man["txm"], args.per_image, args.patch, 2)


if __name__ == "__main__":
    main()
