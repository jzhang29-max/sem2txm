"""Paths, grouping rules and the two scale knobs.

This tool reads the two sibling repos in place rather than copying 2 GB of
micrographs a third time. Point SEM_REPO/TXM_REPO elsewhere with the
SEM2TXM_SEM_REPO / SEM2TXM_TXM_REPO environment variables.
"""
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SEM_REPO = Path(os.environ.get(
    "SEM2TXM_SEM_REPO", "/Users/jiamingzhang/Desktop/sem-crack-detector"))
TXM_REPO = Path(os.environ.get(
    "SEM2TXM_TXM_REPO", "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline"))

SEM_DIR = SEM_REPO / "original"
SEM_MASK_DIR = SEM_REPO / "interior_active_learning" / "paint"
TXM_DIR = TXM_REPO / "images"

CACHE = Path(os.environ.get("SEM2TXM_CACHE", ROOT / "cache"))
FIGURES = ROOT / "figures"
OUT = ROOT / "out"

# Five frames in the SEM set are synthetic app-format tests, not micrographs.
SEM_EXCLUDE_PREFIX = ("apptest_",)

PATCH = 256

# Pixel scale is NOT known for this data: the SEM tool's own provenance records
# `"calibrated": false, "um_per_px": null`. A SEM pixel and a TXM pixel are
# therefore of unknown relative physical size, and everything here is measured
# in pixels. Supply both numbers to resample onto a matched physical scale.
SEM_UM_PER_PX = None
TXM_UM_PER_PX = None


def sem_files():
    fs = sorted(SEM_DIR.glob("*.tif"))
    return [f for f in fs if not f.name.startswith(SEM_EXCLUDE_PREFIX)]


def txm_files():
    return sorted(TXM_DIR.glob("*.tif"))


def txm_stem(path):
    """Average_mosaic_<name>_idx00000_mosaictileAA_img001of010.xrm.bim.bim.tif -> <name>"""
    n = Path(path).name
    n = re.sub(r"^Average_mosaic_", "", n)
    n = re.sub(r"_idx\d+_mosaictile.*$", "", n)
    return n


def sem_group(path):
    """Specimen a SEM frame belongs to, so that train/test can split by specimen
    rather than by frame -- frames of one specimen are not independent."""
    n = Path(path).stem
    m = re.match(r"^\d{6}_(316_(?:H|amb)_b\d+)", n)
    if m:
        return m.group(1).lower()
    m = re.match(r"^MAR_Amb_(AS|Cast|HIP)", n)
    if m:
        return f"mar_amb_{m.group(1).lower()}"
    m = re.match(r"^(AS|Cast|HIP)_24hr", n)
    if m:
        return f"{m.group(1).lower()}_24hr"
    return n.lower()


def txm_group(path):
    """Specimen a TXM mosaic belongs to. Case is normalised because the same
    specimen appears as both B2 and b2 in the filenames."""
    s = txm_stem(path).lower()
    m = re.match(r"^(\d{6})_(hc_316l|wrought_316l|b\d+)", s)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    m = re.match(r"^(\d{6})_(\w+?)_", s)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return s


def summarise():
    from collections import Counter
    sg = Counter(sem_group(f) for f in sem_files())
    tg = Counter(txm_group(f) for f in txm_files())
    print(f"SEM {sum(sg.values())} frames in {len(sg)} specimen groups:")
    for k, v in sorted(sg.items()):
        print(f"   {v:3d}  {k}")
    print(f"TXM {sum(tg.values())} mosaics in {len(tg)} specimen groups:")
    for k, v in sorted(tg.items()):
        print(f"   {v:3d}  {k}")


if __name__ == "__main__":
    summarise()
