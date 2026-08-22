"""A windowed-attention generator, a PatchGAN critic, and the PatchNCE head.

Why this shape. The data is unpaired: no SEM frame is registered to any TXM
mosaic, so there is no per-pixel target to regress against. A plain GAN would be
free to move structure around -- it only has to look like TXM, not like THIS
input -- and a translated crack that has drifted is useless for transferring a
mask. CUT's contrastive term is the fix: a patch of the output must be more
similar to the SAME patch of the input than to any other patch of it, which pins
content in place without ever needing a registered pair. That is the property the
label-transfer experiment depends on, so it is the objective, not the appearance
loss, that is doing the real work here.

The generator bottleneck is windowed multi-head self-attention (shifted between
blocks, as in Swin) rather than the residual convolutions the original CUT used.
Cracks are long, thin and continuous over hundreds of pixels; a stack of 3x3
convolutions reaches that far only by depth, whereas attention inside a 8x8
window at stride 4 relates points 32 px apart in one hop, and the shift carries
that across window borders.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- attention

class WindowAttention(nn.Module):
    def __init__(self, dim, window, heads):
        super().__init__()
        self.dim, self.window, self.heads = dim, window, heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        # Learned relative position bias, (2w-1)^2 distinct offsets per head.
        self.bias_table = nn.Parameter(torch.zeros((2 * window - 1) ** 2, heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)
        coords = torch.stack(torch.meshgrid(
            torch.arange(window), torch.arange(window), indexing="ij")).flatten(1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[..., 0] += window - 1
        rel[..., 1] += window - 1
        rel[..., 0] *= 2 * window - 1
        self.register_buffer("rel_index", rel.sum(-1), persistent=False)

    def forward(self, x, mask=None):
        # x: (nW*B, w*w, C)
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.bias_table[self.rel_index.view(-1)].view(n, n, -1).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.heads, n, n) + mask[None, :, None]
            attn = attn.view(-1, self.heads, n, n)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


def window_partition(x, w):
    b, h, wd, c = x.shape
    x = x.view(b, h // w, w, wd // w, w, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, w * w, c)


def window_reverse(win, w, h, wd):
    b = win.shape[0] // (h * wd // w // w)
    x = win.view(b, h // w, wd // w, w, w, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, wd, -1)


class SwinBlock(nn.Module):
    """LN -> (shifted) window attention -> residual -> LN -> MLP -> residual."""

    def __init__(self, dim, window=8, heads=4, shift=0, mlp_ratio=4.0):
        super().__init__()
        self.window, self.shift = window, shift
        self.n1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window, heads)
        self.n2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self._mask_cache = {}

    def _mask(self, h, w, device):
        """Windows that wrap after the cyclic shift must not attend across the seam."""
        key = (h, w, device.type)
        if key in self._mask_cache:
            return self._mask_cache[key]
        img = torch.zeros((1, h, w, 1), device=device)
        s = self.shift
        cnt = 0
        for hs in (slice(0, -self.window), slice(-self.window, -s), slice(-s, None)):
            for ws in (slice(0, -self.window), slice(-self.window, -s), slice(-s, None)):
                img[:, hs, ws, :] = cnt
                cnt += 1
        win = window_partition(img, self.window).squeeze(-1)
        m = win.unsqueeze(1) - win.unsqueeze(2)
        m = m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)
        self._mask_cache[key] = m
        return m

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        pad_h = (self.window - h % self.window) % self.window
        pad_w = (self.window - w % self.window) % self.window
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, hp, wp = x.shape
        t = x.permute(0, 2, 3, 1)                     # B H W C
        res = t
        t = self.n1(t)
        if self.shift:
            t = torch.roll(t, (-self.shift, -self.shift), dims=(1, 2))
            mask = self._mask(hp, wp, x.device)
        else:
            mask = None
        win = window_partition(t, self.window)
        win = self.attn(win, mask)
        t = window_reverse(win, self.window, hp, wp)
        if self.shift:
            t = torch.roll(t, (self.shift, self.shift), dims=(1, 2))
        t = res + t
        t = t + self.mlp(self.n2(t))
        out = t.permute(0, 3, 1, 2)
        if pad_h or pad_w:
            out = out[:, :, :h, :w]
        return out


# ---------------------------------------------------------------- generator

class Generator(nn.Module):
    """Conv stem, two stride-2 downsamples, `depth` Swin blocks, two upsamples.

    forward(x, layers=[...]) also returns the intermediate activations those
    layer indices name, which is what PatchNCE samples from. Indices are into
    self.taps: 0 stem, 1 down1, 2 down2, 3.. bottleneck blocks.
    """

    def __init__(self, ch=64, depth=6, window=8, heads=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(1, ch, 7), nn.InstanceNorm2d(ch), nn.GELU())
        self.down1 = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.InstanceNorm2d(ch * 2), nn.GELU())
        self.down2 = nn.Sequential(
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1), nn.InstanceNorm2d(ch * 4), nn.GELU())
        self.blocks = nn.ModuleList([
            SwinBlock(ch * 4, window, heads, shift=0 if i % 2 == 0 else window // 2)
            for i in range(depth)])
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch * 4, ch * 2, 3, 1, 1), nn.InstanceNorm2d(ch * 2), nn.GELU())
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch * 2, ch, 3, 1, 1), nn.InstanceNorm2d(ch), nn.GELU())
        self.head = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ch, 1, 7), nn.Tanh())

    def forward(self, x, layers=()):
        feats = []
        h = self.stem(x)
        if 0 in layers:
            feats.append(h)
        h = self.down1(h)
        if 1 in layers:
            feats.append(h)
        h = self.down2(h)
        if 2 in layers:
            feats.append(h)
        encode_only = getattr(self, "_encode_only", False)
        deepest = max(layers) if layers else -1
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if (3 + i) in layers:
                feats.append(h)
            # Encoder-only call (PatchNCE): stop as soon as the deepest requested
            # tap is in hand rather than running the rest of the bottleneck.
            if encode_only and (3 + i) >= deepest:
                return feats
        if encode_only:
            return feats
        h = self.up1(h)
        h = self.up2(h)
        out = self.head(h)
        # Two stride-2 downsamples round the spatial size up to a multiple of 4,
        # so an input of 250 comes back as 252. Harmless for tiled inference (tiles
        # are a power of two) but a silent 2 px shift would misregister a
        # transferred mask, which is the one thing here that must not happen.
        if out.shape[-2:] != x.shape[-2:]:
            out = out[..., :x.shape[-2], :x.shape[-1]]
            if out.shape[-2:] != x.shape[-2:]:
                out = F.pad(out, (0, x.shape[-1] - out.shape[-1],
                                  0, x.shape[-2] - out.shape[-2]), mode="replicate")
        return (out, feats) if layers else out

    def encode(self, x, layers):
        self._encode_only = True
        try:
            return self.forward(x, layers)
        finally:
            self._encode_only = False


# ---------------------------------------------------------------- critic

class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN with spectral norm -- judges local texture, not layout,
    which is what we want: the layout has to come from the input, not the critic."""

    def __init__(self, ch=64, layers=3):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm
        seq = [sn(nn.Conv2d(1, ch, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        m = 1
        for i in range(1, layers):
            m_prev, m = m, min(2 ** i, 8)
            seq += [sn(nn.Conv2d(ch * m_prev, ch * m, 4, 2 if i < layers - 1 else 1, 1)),
                    nn.InstanceNorm2d(ch * m), nn.LeakyReLU(0.2, True)]
        seq += [sn(nn.Conv2d(ch * m, 1, 4, 1, 1))]
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- PatchNCE

class ProjectionHead(nn.Module):
    """Per-tap 2-layer MLP to a common 256-d unit sphere. Built lazily because
    the channel count of each tap is only known at the first forward."""

    def __init__(self, dim=256, n_patches=256):
        super().__init__()
        self.dim, self.n_patches = dim, n_patches
        self.mlps = nn.ModuleDict()

    def forward(self, feats, ids=None):
        out, out_ids = [], []
        for i, f in enumerate(feats):
            b, c, h, w = f.shape
            key = str(i)
            if key not in self.mlps:
                self.mlps[key] = nn.Sequential(
                    nn.Linear(c, self.dim), nn.ReLU(), nn.Linear(self.dim, self.dim)
                ).to(f.device)
            flat = f.permute(0, 2, 3, 1).reshape(b, h * w, c)
            if ids is None:
                n = min(self.n_patches, h * w)
                idx = torch.randperm(h * w, device=f.device)[:n]
            else:
                idx = ids[i]
            sel = flat[:, idx, :]
            z = self.mlps[key](sel.reshape(-1, c))
            out.append(F.normalize(z, dim=1))
            out_ids.append(idx)
        return out, out_ids


def patch_nce_loss(feat_q, feat_k, tau=0.07):
    """InfoNCE where the positive is the SAME spatial location in the other image
    and the negatives are the other sampled locations of that same image."""
    n, c = feat_q.shape
    pos = (feat_q * feat_k).sum(dim=1, keepdim=True) / tau
    neg = (feat_q @ feat_k.t()) / tau
    eye = torch.eye(n, dtype=torch.bool, device=feat_q.device)
    neg = neg.masked_fill(eye, -1e4)
    logits = torch.cat([pos, neg], dim=1)
    target = torch.zeros(n, dtype=torch.long, device=feat_q.device)
    return F.cross_entropy(logits, target)
