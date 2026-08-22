"""Look at both modalities side by side before building anything.

Writes a contact sheet of random patches from each domain, so that whatever the
translator learns can be compared against what the source data actually looks
like. Also reports how much of each TXM mosaic is padding, because the mosaics
are stitched and the padding is not specimen.
"""
import glob
import os

import numpy as np
import tifffile
from PIL import Image

SEM_DIR = "/Users/jiamingzhang/Desktop/sem-crack-detector/original"
TXM_DIR = "/Users/jiamingzhang/Desktop/TXM_Crack_Detection_Pipeline/images"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

# The SEM set carries five synthetic app-test frames that are not micrographs.
SEM_EXCLUDE = ("apptest_",)


def sem_files():
    fs = sorted(glob.glob(os.path.join(SEM_DIR, "*.tif")))
    return [f for f in fs if not os.path.basename(f).startswith(SEM_EXCLUDE)]


def txm_files():
    return sorted(glob.glob(os.path.join(TXM_DIR, "*.tif")))


def stretch(a, lo=1, hi=99):
    """Percentile stretch to uint8, for display only."""
    a = a.astype(np.float32)
    p1, p99 = np.percentile(a, [lo, hi])
    if p99 <= p1:
        return np.zeros(a.shape, np.uint8)
    return np.clip((a - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)


def padding_fraction(a):
    """TXM mosaics pad the area outside the tile grid with (near-)zero."""
    return float((a <= 1e-6).mean())


def contact_sheet(files, n, patch, seed, label, reader):
    rng = np.random.default_rng(seed)
    tiles = []
    picks = rng.choice(len(files), size=min(n, len(files)), replace=False)
    for i in picks:
        a = reader(files[i])
        if a.ndim == 3:
            a = a.mean(axis=2)
        h, w = a.shape
        if h < patch or w < patch:
            continue
        # Reject patches that are mostly mosaic padding.
        for _ in range(40):
            y = int(rng.integers(0, h - patch))
            x = int(rng.integers(0, w - patch))
            p = a[y:y + patch, x:x + patch]
            if padding_fraction(p) < 0.02:
                break
        tiles.append((stretch(p), os.path.basename(files[i])))
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * patch, cols * patch), np.uint8)
    for k, (t, _) in enumerate(tiles):
        r, c = divmod(k, cols)
        sheet[r * patch:(r + 1) * patch, c * patch:(c + 1) * patch] = t
    Image.fromarray(sheet).save(os.path.join(OUT, f"contact_{label}.png"))
    print(f"wrote figures/contact_{label}.png  ({len(tiles)} patches of {patch}px)")
    for _, nm in tiles[:6]:
        print("   ", nm[:70])


def main():
    os.makedirs(OUT, exist_ok=True)
    sf, tf = sem_files(), txm_files()
    print(f"SEM micrographs: {len(sf)}   TXM mosaics: {len(tf)}")

    print("\nTXM padding fraction (how much of each mosaic is not specimen):")
    fracs = []
    for f in tf[:12]:
        a = tifffile.imread(f)
        fr = padding_fraction(a)
        fracs.append(fr)
        print(f"  {fr:5.1%}  {os.path.basename(f)[:62]}")
    print(f"  mean over {len(fracs)} sampled: {np.mean(fracs):.1%}")

    contact_sheet(sf, 8, 512, 0, "sem", tifffile.imread)
    contact_sheet(tf, 8, 512, 0, "txm", tifffile.imread)


if __name__ == "__main__":
    main()
