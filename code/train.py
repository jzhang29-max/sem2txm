"""Train the SEM -> TXM translator (CUT objective, windowed-attention generator).

Three terms, and it is worth being clear about what each one buys:

  adversarial   makes the output look like a flat-fielded TXM mosaic
  PatchNCE      makes the output stay where the input put it
  identity NCE  G(real TXM) ~ real TXM, which stops the generator from applying
                its transform to material that is already in the target domain

The middle term is the one the downstream label transfer rests on, so a
structure-retention proxy is logged every step alongside the losses: the
correlation between the input's edge map and the output's. If that falls while
the GAN loss improves, the model is buying appearance with geometry and the run
is not usable no matter how good the samples look.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config as C
from model import Generator, PatchDiscriminator, ProjectionHead, patch_nce_loss

NCE_LAYERS = (0, 1, 2, 4, 6)


def _tap_channels(ch, depth):
    """Channel count at each NCE tap, in NCE_LAYERS order -- needed to build the
    projection heads before loading their weights on resume."""
    per = {0: ch, 1: ch * 2, 2: ch * 4}
    return [per.get(l, ch * 4) for l in NCE_LAYERS]


def device_of(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Bank:
    """Patch bank on disk; uint8 in [0,255] -> float in [-1,1], with flips and
    90-degree rotations. Both modalities are direction-bearing (SEM has polishing
    scratches, TXM has a residual tile axis) so the dihedral group is the honest
    limit of what can be augmented without inventing anisotropy."""

    def __init__(self, path, rng):
        self.a = np.load(path, mmap_mode="r")
        self.rng = rng

    def __len__(self):
        return len(self.a)

    def batch(self, n):
        idx = self.rng.integers(0, len(self.a), size=n)
        out = np.empty((n, 1, self.a.shape[1], self.a.shape[2]), np.float32)
        for j, i in enumerate(idx):
            p = np.asarray(self.a[i], np.float32) / 127.5 - 1.0
            k = int(self.rng.integers(0, 4))
            if k:
                p = np.rot90(p, k)
            if self.rng.random() < 0.5:
                p = p[:, ::-1]
            out[j, 0] = p
        return torch.from_numpy(out)


def grad_xy(t):
    """Forward differences, cropped to a common shape."""
    gx = t[..., :, 1:] - t[..., :, :-1]
    gy = t[..., 1:, :] - t[..., :-1, :]
    return gx[..., 1:, :], gy[..., :, 1:]


def idt_pixel_stats(idt, real):
    """Distortion the identity branch actually inflicts, in pixel space.

    Logged every step alongside the losses because the run that produced the 4 px
    blur equivalent showed a healthy identity-NCE curve the whole way. Whatever is
    not measured here is not known.
    """
    a = idt.detach().flatten(1)
    b = real.detach().flatten(1)
    l1 = (a - b).abs().mean().item()
    am = a - a.mean(1, keepdim=True)
    bm = b - b.mean(1, keepdim=True)
    r = ((am * bm).sum(1) / (am.norm(dim=1) * bm.norm(dim=1) + 1e-8)).mean().item()
    return l1, r


def edge_corr(a, b):
    """Correlation of gradient magnitudes between input and output.

    A cheap proxy, and a CONFOUNDED one -- read it as a diagnostic, not a verdict.

    The confound is not the one first assumed. The guess was that TXM carries less
    high-frequency content than SEM, so shedding fine edges would be correct
    behaviour. The measured spectra say the opposite: in the four finest bands real
    TXM holds 0.0188 / 0.0053 / 0.0024 / 0.0019 of its power against SEM's 0.0155 /
    0.0042 / 0.0014 / 0.0005. TXM is NOISIER at fine scale, not smoother -- X-ray
    photon noise.

    So the real confound is that matching TXM means ADDING fine-scale noise, and
    noise is by construction uncorrelated with the input's own fine detail. That
    pulls a gradient-map correlation down while coarse structure is untouched.
    Either way the number is ambiguous, but for the opposite reason.

    The unconfounded test of structure retention is crack-contrast retention in
    eval_translation.py: whether a hand-marked crack is still darker than its own
    local surroundings after translation. That is scale-free, and it is what label
    transfer actually depends on. This number is logged because it is nearly free
    and a collapse to ~0 would be worth catching early.
    """
    def grad(t):
        gx = t[..., :, 1:] - t[..., :, :-1]
        gy = t[..., 1:, :] - t[..., :-1, :]
        return (gx[..., 1:, :].abs() + gy[..., :, 1:].abs()).flatten(1)
    ga, gb = grad(a), grad(b)
    ga = ga - ga.mean(1, keepdim=True)
    gb = gb - gb.mean(1, keepdim=True)
    num = (ga * gb).sum(1)
    den = ga.norm(dim=1) * gb.norm(dim=1) + 1e-8
    return (num / den).mean().item()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--lambda-nce", type=float, default=1.0)
    ap.add_argument("--lambda-idt-l1", type=float, default=0.0,
                    help="pixel L1 on the identity branch, |G(y)-y|. The NCE "
                         "identity term operates in feature space and demonstrably "
                         "does not bound pixel distortion: its loss reached 0.024 "
                         "while G(y) still lost information equivalent to a 4 px "
                         "blur on held-out frames.")
    ap.add_argument("--lambda-idt-hf", type=float, default=0.0,
                    help="L1 on image GRADIENTS of the identity branch. Aimed at "
                         "the measured failure specifically -- what G loses is fine "
                         "detail, and a plain L1 is dominated by low frequencies "
                         "that were never the problem.")
    ap.add_argument("--lambda-gan", type=float, default=1.0)
    ap.add_argument("--nce-patches", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(C.ROOT / "runs" / "cut"))
    ap.add_argument("--resume", default="", help="checkpoint to continue from")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--sample-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=2000)
    args = ap.parse_args()

    dev = device_of(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args) | {"device": str(dev)}, indent=1))

    sem = Bank(C.CACHE / "bank_sem.npy", rng)
    txm = Bank(C.CACHE / "bank_txm.npy", rng)
    print(f"device={dev}  SEM patches={len(sem)}  TXM patches={len(txm)}")

    G = Generator(args.ch, args.depth, args.window).to(dev)
    D = PatchDiscriminator(args.ch).to(dev)
    H = ProjectionHead(n_patches=args.nce_patches).to(dev)
    nparam = sum(p.numel() for p in G.parameters())
    print(f"generator parameters: {nparam/1e6:.2f} M")

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_h = None                      # built after the heads exist (lazy channels)

    # Resume exists because this machine sleeps when idle: a run that logged
    # 1.57 s/iter for 250 iterations then showed 197 s/iter was not throttling,
    # it was the host suspending with the wall clock still running. Checkpoints
    # are frequent and continuing is cheaper than restarting.
    start = 1
    if args.resume:
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"])
        H(  [torch.zeros(1, c, 8, 8, device=dev)
             for c in _tap_channels(args.ch, args.depth)] )   # build lazy heads
        H.load_state_dict(ck["H"])
        start = int(ck["iter"]) + 1
        print(f"resumed from {args.resume} at iter {ck['iter']}")

    mode = "a" if (args.resume and (out / "log.csv").exists()) else "w"
    log = open(out / "log.csv", mode, newline="")
    w = csv.writer(log)
    if mode == "w":
        w.writerow(["iter", "d_loss", "g_gan", "g_nce", "g_idt", "g_idt_l1",
                "g_idt_hf", "idt_px_l1", "idt_px_r", "xlate_l1", "edge_corr",
                "sec"])

    t0 = time.time()
    for it in range(start, args.iters + 1):
        # Linear lr decay over the final third.
        if it > args.iters * 2 // 3:
            frac = 1.0 - (it - args.iters * 2 // 3) / (args.iters / 3 + 1e-9)
            for o in (opt_g, opt_d) + ((opt_h,) if opt_h else ()):
                for g in o.param_groups:
                    g["lr"] = args.lr * max(frac, 0.0)

        a = sem.batch(args.batch).to(dev)      # SEM  (source)
        b = txm.batch(args.batch).to(dev)      # TXM  (target)

        # One generator pass over both domains, as CUT does: the second half is
        # the identity branch. The encoder taps PatchNCE needs for the two INPUTS
        # are taken from this same pass and split, rather than re-encoding a and b
        # separately -- that removes two of the four encoder passes per step
        # (measured 1.93 -> 1.45 s/iter at batch 8 on an M4 Max).
        both = torch.cat([a, b], 0)
        fake_both, feats_both = G(both, NCE_LAYERS)
        fake_b = fake_both[:args.batch]         # translated SEM
        idt_b = fake_both[args.batch:]          # G(TXM), should be ~TXM
        f_a = [f[:args.batch] for f in feats_both]
        f_b = [f[args.batch:] for f in feats_both]

        # ---- critic
        for p in D.parameters():
            p.requires_grad_(True)
        opt_d.zero_grad(set_to_none=True)
        pred_fake = D(fake_b.detach())
        pred_real = D(b)
        d_loss = 0.5 * (F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
                        + F.mse_loss(pred_real, torch.ones_like(pred_real)))
        d_loss.backward()
        opt_d.step()

        # ---- generator
        for p in D.parameters():
            p.requires_grad_(False)
        opt_g.zero_grad(set_to_none=True)
        pred = D(fake_b)
        g_gan = args.lambda_gan * F.mse_loss(pred, torch.ones_like(pred))

        f_fb = G.encode(fake_b, NCE_LAYERS)
        z_a, ids = H(f_a)
        z_fb, _ = H(f_fb, ids)
        g_nce = sum(patch_nce_loss(q, k) for q, k in zip(z_fb, z_a)) / len(z_a)
        g_nce = args.lambda_nce * g_nce

        f_ib = G.encode(idt_b, NCE_LAYERS)
        z_b, ids_b = H(f_b)
        z_ib, _ = H(f_ib, ids_b)
        g_idt = sum(patch_nce_loss(q, k) for q, k in zip(z_ib, z_b)) / len(z_b)
        g_idt = args.lambda_nce * g_idt

        # Pixel-space identity terms. The target here is the input itself, so
        # there is no ambiguity for L1 to blur away -- it penalises distortion
        # directly. The gradient term is separate because a plain L1 over these
        # images is dominated by the low frequencies, and the measured loss was in
        # the high ones.
        g_idt_l1 = torch.zeros((), device=dev)
        g_idt_hf = torch.zeros((), device=dev)
        if args.lambda_idt_l1 > 0:
            g_idt_l1 = args.lambda_idt_l1 * F.l1_loss(idt_b, b)
        if args.lambda_idt_hf > 0:
            ix, iy = grad_xy(idt_b)
            bx, by = grad_xy(b)
            g_idt_hf = args.lambda_idt_hf * (F.l1_loss(ix, bx) + F.l1_loss(iy, by))

        (g_gan + g_nce + g_idt + g_idt_l1 + g_idt_hf).backward()
        if opt_h is None:
            opt_h = torch.optim.Adam(H.parameters(), lr=args.lr, betas=(0.5, 0.999))
        opt_g.step()
        opt_h.step()
        opt_h.zero_grad(set_to_none=True)

        if it % args.log_every == 0 or it == 1:
            ec = edge_corr(a.detach(), fake_b.detach())
            px_l1, px_r = idt_pixel_stats(idt_b, b)
            # Collapse guard: a pixel identity term pushes G toward the identity
            # map, and G = identity satisfies it perfectly while translating
            # nothing. This is how much G moves its SEM input; if it approaches 0
            # the run has bought fidelity by abandoning the task.
            xl1 = (fake_b.detach() - a.detach()).abs().mean().item()
            w.writerow([it, f"{d_loss.item():.4f}", f"{g_gan.item():.4f}",
                        f"{g_nce.item():.4f}", f"{g_idt.item():.4f}",
                        f"{float(g_idt_l1):.4f}", f"{float(g_idt_hf):.4f}",
                        f"{px_l1:.4f}", f"{px_r:.4f}", f"{xl1:.4f}",
                        f"{ec:.4f}", f"{time.time()-t0:.1f}"])
            log.flush()
            print(f"it {it:6d}  D {d_loss.item():.3f}  Ggan {g_gan.item():.3f}  "
                  f"NCE {g_nce.item():.3f}  idt {g_idt.item():.3f}  "
                  f"idtL1 {float(g_idt_l1):.3f}  idtHF {float(g_idt_hf):.3f}  "
                  f"| px_l1 {px_l1:.4f} px_r {px_r:+.3f} xlate {xl1:.4f} | "
                  f"edge_corr {ec:.3f}  {time.time()-t0:.0f}s", flush=True)

        if it % args.sample_every == 0 or it == args.iters:
            save_samples(out / "samples" / f"{it:06d}.png", a, fake_b, b, idt_b)
        if it % args.save_every == 0 or it == args.iters:
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "H": H.state_dict(),
                        "args": vars(args), "iter": it}, out / "ckpt.pt")

    log.close()
    print(f"done in {(time.time()-t0)/60:.1f} min -> {out}")


def save_samples(path, a, fake_b, b, idt_b):
    from PIL import Image
    def to8(t):
        x = ((t.detach().cpu().numpy()[:, 0] + 1) * 127.5).clip(0, 255).astype(np.uint8)
        return x
    rows = [to8(a), to8(fake_b), to8(b), to8(idt_b)]
    n = min(4, rows[0].shape[0])
    p = rows[0].shape[1]
    sheet = np.zeros((len(rows) * p, n * p), np.uint8)
    for r, row in enumerate(rows):
        for c in range(n):
            sheet[r * p:(r + 1) * p, c * p:(c + 1) * p] = row[c]
    Image.fromarray(sheet).save(path)


if __name__ == "__main__":
    main()
