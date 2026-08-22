"""Image panels: what the preprocessing does, and what a translation looks like.

The real-TXM column is a REFERENCE, not a target. No SEM frame is registered to
any TXM mosaic, so the right-hand image is a different field of view of the same
material -- it is there to show what the destination domain looks like, and
nothing in it should be read as the correct answer for the row it sits in.
"""
import argparse
import json
import sys

import numpy as np
import tifffile
from PIL import Image, ImageDraw

import config as C

sys.path.insert(0, str(C.TXM_REPO / "code"))


def to8(a):
    a = np.asarray(a, np.float32)
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-8) * 255, 0, 255).astype(np.uint8)


def label_strip(w, text, h=22):
    im = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(im)
    d.text((4, 5), text, fill=0)
    return np.array(im)


def grid(cells, titles, gap=6):
    """cells: list of rows, each a list of 2-D uint8 arrays of equal size."""
    ph, pw = cells[0][0].shape
    ncol = len(cells[0])
    head = 22
    W = ncol * pw + (ncol - 1) * gap
    H = head + len(cells) * ph + (len(cells) - 1) * gap
    sheet = np.full((H, W), 255, np.uint8)
    for c in range(ncol):
        x = c * (pw + gap)
        sheet[:head, x:x + pw] = label_strip(pw, titles[c])
    for r, row in enumerate(cells):
        y = head + r * (ph + gap)
        for c, cell in enumerate(row):
            x = c * (pw + gap)
            sheet[y:y + ph, x:x + pw] = cell
    return Image.fromarray(sheet)


def preprocessing_panel(size=512, seed=3):
    """Raw TXM vs the flat-fielded view the translator actually targets."""
    import destitch, flatfield
    rng = np.random.default_rng(seed)
    man = json.loads((C.CACHE / "manifest.json").read_text())
    picks = [e for e in man["txm"] if not e["reference"]][:40]
    rows, used = [], []
    for e in rng.permutation(len(picks))[:3]:
        ent = picks[int(e)]
        src = [f for f in C.txm_files() if C.txm_stem(f) == ent["stem"]]
        if not src:
            continue
        raw = tifffile.imread(str(src[0])).astype(np.float32)
        ff = np.load(C.CACHE / "txm" / f"{ent['stem']}.npy")
        h, w = ff.shape
        if h < size or w < size:
            continue
        for _ in range(60):
            y = int(rng.integers(0, h - size)); x = int(rng.integers(0, w - size))
            if (ff[y:y + size, x:x + size] == 0).mean() < 0.01:
                break
        rows.append([to8(raw[y:y + size, x:x + size]), ff[y:y + size, x:x + size]])
        used.append(ent["stem"])
        if len(rows) == 3:
            break
    if not rows:
        return
    im = grid(rows, ["raw TXM mosaic", "destitched + flat-fielded"])
    im.save(C.FIGURES / "preprocessing.png")
    print("  wrote figures/preprocessing.png", used)


def translation_panel(ckpt, size=512, n=4, seed=7, device="auto"):
    import torch
    from train import device_of
    from translate import load_generator
    dev = device_of(device)
    G, ck = load_generator(ckpt, dev)
    rng = np.random.default_rng(seed)

    sem = np.load(C.CACHE / "bank_sem.npy", mmap_mode="r")
    txm = np.load(C.CACHE / "bank_txm.npy", mmap_mode="r")
    si = rng.choice(len(sem), n, replace=False)
    ti = rng.choice(len(txm), n, replace=False)
    src = np.asarray(sem[np.sort(si)], np.float32) / 255.0

    with torch.no_grad():
        t = torch.from_numpy(src[:, None] * 2 - 1).to(dev)
        out = ((G(t).cpu().numpy()[:, 0] + 1) * 0.5)

    rows = []
    for k in range(n):
        rows.append([(src[k] * 255).astype(np.uint8),
                     (out[k].clip(0, 1) * 255).astype(np.uint8),
                     np.asarray(txm[np.sort(ti)[k]], np.uint8)])
    im = grid(rows, ["SEM input", "predicted TXM", "real TXM (reference, unrelated field)"])
    im.save(C.FIGURES / "translation_examples.png")
    print(f"  wrote figures/translation_examples.png (iter {ck['iter']})")


def full_frame_panel(ckpt, stem=None, scale=6, device="auto"):
    """A whole micrograph translated, downsampled for the README."""
    from train import device_of
    from translate import load_generator, translate
    dev = device_of(device)
    G, ck = load_generator(ckpt, dev)
    man = json.loads((C.CACHE / "manifest.json").read_text())
    cand = [e["stem"] for e in man["sem"]
            if (C.SEM_MASK_DIR / f"{e['stem']}_correction_mask.png").exists()]
    stem = stem or (cand[0] if cand else man["sem"][0]["stem"])
    img = np.load(C.CACHE / "sem" / f"{stem}.npy").astype(np.float32) / 255.0
    out = translate(G, img, dev)

    def ds(a):
        p = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
        return np.array(p.resize((p.width // scale, p.height // scale), Image.LANCZOS))
    a, b = ds(img), ds(out)
    im = grid([[a, b]], ["SEM input (full frame)", "predicted TXM"])
    im.save(C.FIGURES / "full_frame.png")
    print(f"  wrote figures/full_frame.png  {stem} {img.shape} (iter {ck['iter']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(C.ROOT / "runs" / "cut" / "ckpt.pt"))
    ap.add_argument("--only", default="", help="preprocessing | translation | full")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    print("panels:")
    if args.only in ("", "preprocessing"):
        preprocessing_panel()
    from pathlib import Path
    if not Path(args.ckpt).exists():
        print("  (no checkpoint yet -- skipping translation panels)")
        return
    if args.only in ("", "translation"):
        translation_panel(args.ckpt, device=args.device)
    if args.only in ("", "full"):
        full_frame_panel(args.ckpt, device=args.device)


if __name__ == "__main__":
    main()
