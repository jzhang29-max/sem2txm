"""Apply a trained translator to whole micrographs.

The generator is trained on 256 px patches but the frames are up to 27 MP, so
inference is tiled. Tiles are blended with a raised-cosine window rather than
butted together: a hard join puts a step edge in the output, and a step edge is
exactly what a crack detector downstream would fire on. Overlap defaults to a
quarter tile, which is wider than the generator's effective receptive field at
the join.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import config as C
from model import Generator


def load_generator(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ck["args"]
    G = Generator(a["ch"], a["depth"], a["window"]).to(device)
    G.load_state_dict(ck["G"])
    G.eval()
    return G, ck


def _blend_window(tile, overlap):
    """Raised cosine that reaches 0 at the tile edge, 1 by `overlap` pixels in."""
    r = np.ones(tile)
    ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(overlap) + 0.5) / overlap))
    r[:overlap] = ramp
    r[-overlap:] = ramp[::-1]
    return np.outer(r, r).astype(np.float32)


@torch.no_grad()
def translate(G, img01, device, tile=256, overlap=64, batch=8):
    """img01: float32 HxW in [0,1] (already flat-fielded). Returns same shape in [0,1]."""
    h, w = img01.shape
    step = tile - overlap
    ph = max(tile, int(np.ceil((h - overlap) / step)) * step + overlap)
    pw = max(tile, int(np.ceil((w - overlap) / step)) * step + overlap)
    pad = np.pad(img01, ((0, ph - h), (0, pw - w)), mode="reflect")
    acc = np.zeros((ph, pw), np.float32)
    wsum = np.zeros((ph, pw), np.float32)
    win = _blend_window(tile, overlap)

    coords = [(y, x) for y in range(0, ph - tile + 1, step)
              for x in range(0, pw - tile + 1, step)]
    for i in range(0, len(coords), batch):
        chunk = coords[i:i + batch]
        stack = np.stack([pad[y:y + tile, x:x + tile] for y, x in chunk])[:, None]
        t = torch.from_numpy(stack * 2.0 - 1.0).to(device)
        o = G(t).cpu().numpy()[:, 0]
        o = (o + 1.0) * 0.5
        for (y, x), tileout in zip(chunk, o):
            acc[y:y + tile, x:x + tile] += tileout * win
            wsum[y:y + tile, x:x + tile] += win
    out = acc / np.clip(wsum, 1e-6, None)
    return np.clip(out[:h, :w], 0, 1)


def write_pair_png(path, src, out, scale=4):
    """SEM input beside its translation, downsampled so a 27 MP frame is viewable."""
    from PIL import Image
    def ds(a):
        im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
        if scale > 1:
            im = im.resize((max(1, im.width // scale), max(1, im.height // scale)),
                           Image.LANCZOS)
        return np.array(im)
    a, b = ds(src), ds(out)
    gap = 8
    sheet = np.full((a.shape[0], a.shape[1] + gap + b.shape[1]), 255, np.uint8)
    sheet[:, :a.shape[1]] = a
    sheet[:, a.shape[1] + gap:] = b
    Image.fromarray(sheet).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(C.ROOT / "runs" / "cut" / "ckpt.pt"))
    ap.add_argument("--images", nargs="*", help="SEM stems (default: all cached)")
    ap.add_argument("--out", default=str(C.CACHE / "translated"))
    ap.add_argument("--tile", type=int, default=512,
                    help="inference tile; larger is fewer, bigger GPU ops for the "
                         "same pixels. Training used 256 but the generator is "
                         "size-agnostic (conv + windowed attention pads to the "
                         "window).")
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="retranslate frames already on disk")
    ap.add_argument("--masked-only", action="store_true",
                    help="only frames that carry a hand-drawn correction mask -- "
                         "the 39 the label-transfer experiment can actually use")
    ap.add_argument("--png", action="store_true",
                    help="also write a viewable side-by-side PNG per frame")
    ap.add_argument("--png-scale", type=int, default=4,
                    help="downsample factor for the PNG (originals are up to 27 MP)")
    args = ap.parse_args()

    from train import device_of
    dev = device_of(args.device)
    G, ck = load_generator(args.ckpt, dev)
    print(f"generator from iter {ck['iter']} on {dev}")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    stems = args.images or [p.stem for p in sorted((C.CACHE / "sem").glob("*.npy"))
                            if not p.name.endswith(".box.npy")]
    if args.masked_only:
        stems = [s for s in stems
                 if (C.SEM_MASK_DIR / f"{s}_correction_mask.png").exists()]
        print(f"{len(stems)} frames carry a correction mask")
    if args.limit:
        stems = stems[:args.limit]
    for i, stem in enumerate(stems):
        src = C.CACHE / "sem" / f"{stem}.npy"
        if not src.exists():
            print(f"  skip {stem}: not cached")
            continue
        if not args.force and (outdir / f"{stem}.npy").exists():
            print(f"  [{i+1}/{len(stems)}] {stem[:52]} already done")
            continue
        img01 = np.load(src).astype(np.float32) / 255.0
        o = translate(G, img01, dev, args.tile, args.overlap, args.batch)
        np.save(outdir / f"{stem}.npy", (o * 255).astype(np.uint8))
        if args.png:
            write_pair_png(outdir / f"{stem}.png", img01, o, args.png_scale)
        print(f"  [{i+1}/{len(stems)}] {stem[:52]} {img01.shape} "
              f"-> median {np.median(o):.3f}", flush=True)
    print(f"translated -> {outdir}")


if __name__ == "__main__":
    main()
