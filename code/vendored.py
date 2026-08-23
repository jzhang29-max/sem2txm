"""Verbatim copies of the three sibling-repo functions the PREDICT path needs.

Why this file exists. Everything else here imports its preprocessing from the two
sibling repositories, deliberately, so the target domain is exactly the domain those
tools work in. But that made `./run predict` -- the first command the README tells a
reader to run -- fail on any machine that had not also cloned both siblings:

    ModuleNotFoundError: No module named 'flatfield'

A standalone predictor should not need two other repositories. So these three
functions are copied here VERBATIM, and `predict.py` prefers the sibling versions
when they are importable and falls back to these when they are not. The equivalence
is asserted by `selftest()` below, not assumed: if a sibling repo is present, it
checks the two paths agree to float32 on real data.

Sources, copied unmodified:
  _detect_databar_top, find_field_of_view
      <sem-crack-detector>/code/detect_cracks.py
  flatfield
      <TXM_Crack_Detection_Pipeline>/code/flatfield.py

If those upstream functions change, this copy goes stale silently in the fallback
case. `selftest()` is what catches that, and `./run predict --selftest` runs it.
"""
import numpy as np
from scipy.ndimage import gaussian_filter


def _detect_databar_top(img8, search_frac=0.25, window=10, std_ratio=0.4, mean_ratio=0.85):
    """
    Some SEM exports (e.g. the raw instrument captures in this dataset) have
    an info bar burned in at the bottom (scale bar, detector, voltage, ...).
    Returns the first row of that bar, or h if there is none.

    Two independent signals, and the higher (earlier) one wins, because either
    alone leaves real bars in the frame.

    1. STATISTICAL. A databar background is close to solid, so both the row-wise
       mean brightness AND the row-wise standard deviation drop sharply at the
       transition. Comparing each row against the max over the preceding `window`
       rows keeps this stable against ordinary row-to-row noise. (An earlier
       version looked for a spike in horizontal gradient energy instead, which
       assumed the specimen surface is smooth -- false on busy, high-contrast
       captures where grain texture has edge energy as high as the bar.)

    2. GEOMETRIC. The panel is a RECTANGLE, so its top edge changes brightness
       across essentially the whole width at a single row -- unlike a jagged
       specimen boundary or a dark void, which change locally.

    Signal 1 alone missed five captures here (MAR_Amb_Cast_CBS_0002/0005,
    MAR_Amb_Cast_ETD_0002/0005, MAR_Amb_HIP_CBS_0003): their panel is a flat
    MID-GREY, not dark enough for mean_ratio to trip, so 240 rows of bar stayed in
    frame and the model detected the bar's own text as cracks -- 243,405 red pixels
    on one of them. Signal 2 catches all five. Measured across all 62 images, the
    60% width threshold moves the crop on exactly those five and nothing else.
    """
    h, w = img8.shape
    search_start = int(h * (1 - search_frac))
    rows = img8[search_start:].astype(np.float32)
    if rows.shape[0] < window + 5:
        return h

    bar_top_stat = h
    row_means = rows.mean(axis=1)
    row_stds = rows.std(axis=1)
    for i in range(window, len(row_means)):
        baseline_std = row_stds[i - window:i].max()
        baseline_mean = row_means[i - window:i].max()
        if (baseline_std > 3 and row_stds[i] < baseline_std * std_ratio
                and row_means[i] < baseline_mean * mean_ratio):
            bar_top_stat = search_start + i
            break

    bar_top_edge = h
    best_frac, best_row = 0.0, None
    for i in range(1, rows.shape[0]):
        frac = float((np.abs(rows[i] - rows[i - 1]) > 10).mean())
        if frac > best_frac:
            best_frac, best_row = frac, search_start + i
    # >=20 rows below it, so a near-bottom noise spike cannot masquerade as a panel
    if best_frac >= 0.60 and best_row is not None and h - best_row >= 20:
        bar_top_edge = best_row

    return min(bar_top_stat, bar_top_edge)

def find_field_of_view(img8, bright_thresh=12, shrink_frac=0.16, min_keep_frac=0.2):
    """
    Auto-detect the usable sample area, excluding any burned-in info bar and
    any circular aperture vignette (dark corners around a round field of
    view) -- both are common in raw SEM exports and would otherwise be
    misread as one giant 'crack' by the darkness threshold. Images that are
    already tightly cropped (no bar, no vignette) pass through unchanged.
    Returns (x0, y0, x1, y1).
    """
    h, w = img8.shape
    bar_top = _detect_databar_top(img8)
    workspace = img8[:bar_top].astype(np.float32)
    row_ok = np.where(workspace.mean(axis=1) > bright_thresh)[0]
    col_ok = np.where(workspace.mean(axis=0) > bright_thresh)[0]
    if len(row_ok) == 0 or len(col_ok) == 0:
        return 0, 0, w, bar_top

    y0, y1 = int(row_ok.min()), int(row_ok.max())
    x0, x1 = int(col_ok.min()), int(col_ok.max())
    fills_workspace = (x1 - x0) > 0.97 * w and (y1 - y0) > 0.97 * bar_top

    # A genuine circular aperture vignette darkens all FOUR corners
    # symmetrically (that's what "round field of view" means). A large but
    # ordinary dark REGION -- a sample edge, a void, a low-signal patch near
    # one side -- can just as easily pull the row/col brightness bounding
    # box in from one or two sides without being vignetting at all. Checked
    # directly on this dataset: of the images that failed the fills_workspace
    # check, none had all four corners dark in a way consistent with a round
    # cutoff (usually only one or two corners, or a band along one edge) --
    # confirmed by looking at actual corner brightness, not inferred. Require
    # that signature explicitly before committing to the inscribed-square
    # crop; otherwise this dataset's real dark regions get mistaken for
    # vignetting and lose 50-70% of a perfectly good frame.
    corner_size = max(20, int(0.03 * min(w, bar_top)))
    corners = [
        workspace[:corner_size, :corner_size].mean(),
        workspace[:corner_size, -corner_size:].mean(),
        workspace[-corner_size:, :corner_size].mean(),
        workspace[-corner_size:, -corner_size:].mean(),
    ]
    looks_like_round_vignette = all(c < bright_thresh for c in corners)

    if fills_workspace or not looks_like_round_vignette:
        # No vignette (or not the round kind this crop assumes) -- exclude
        # only the databar, keep the rest of the frame as-is.
        x0, y0, x1, y1 = 0, 0, w, bar_top
    else:
        # Treat the bright blob as a circle (its bounding box can be
        # asymmetric if the bar clipped one side) and crop to the largest
        # square inscribed in that circle. A per-axis shrink can still leave
        # a corner poking into the vignette when the box isn't square (e.g.
        # the bar ate into the height but not the width); an inscribed
        # square is geometrically guaranteed to clear every corner.
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        radius = max(y1 - y0, x1 - x0) / 2
        half_side = radius / math.sqrt(2) * (1 - shrink_frac)
        y0, y1 = int(cy - half_side), int(cy + half_side)
        x0, x1 = int(cx - half_side), int(cx + half_side)
        y0, x0 = max(y0, 0), max(x0, 0)
        y1, x1 = min(y1, bar_top), min(x1, w)

    if (x1 - x0) * (y1 - y0) < min_keep_frac * w * h:
        return 0, 0, w, bar_top  # detection looked wrong -- fall back safely
    return x0, y0, x1, y1

def flatfield(img, sigma_y=16.0, sigma_x=22.0, blank_frac=0.05):
    """Divide by a blurred background estimated only from valid (non-blank) pixels."""
    img = img.astype(np.float64)
    ref = np.median(img[img > 0]) if np.any(img > 0) else 1.0
    valid = (img >= max(blank_frac * ref, 1e-6)).astype(np.float64)

    num = gaussian_filter(img * valid, sigma=(sigma_y, sigma_x), mode="nearest")
    den = gaussian_filter(valid, sigma=(sigma_y, sigma_x), mode="nearest")
    blur = num / np.clip(den, 1e-3, None)

    out = img / np.clip(blur, max(1e-3 * ref, 1e-9), None)
    out[valid == 0] = 0.0
    return out

def selftest(verbose=True):
    """Assert the vendored copies match the sibling originals, on real data.

    Only meaningful where the siblings are present. Returns True if they agree, None
    if the originals are not importable (a legitimate standalone install), False if
    they disagree -- which means this file has gone stale against upstream.
    """
    import sys
    from pathlib import Path
    import config as C
    for p in (C.SEM_REPO / "code", C.TXM_REPO / "code"):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from detect_cracks import find_field_of_view as up_fov
        import flatfield as up_ff
    except Exception:
        if verbose:
            print("  sibling repos not importable -- nothing to compare against")
        return None

    rng = np.random.default_rng(0)
    ok = True
    # a synthetic frame with a dark panel along the bottom, like the real captures
    img = (rng.random((600, 900)) * 90 + 140).astype(np.uint8)
    img[520:, :] = 40
    a, b = find_field_of_view(img), up_fov(img)
    if a != b:
        ok = False
        print(f"  MISMATCH find_field_of_view: vendored {a} vs upstream {b}")
    elif verbose:
        print(f"  find_field_of_view agrees: {a}")

    f = (rng.random((300, 400)) * 0.2 + 1.0).astype(np.float32)
    x, y = flatfield(f), up_ff.flatfield(f)
    if isinstance(y, tuple):
        y = y[0]
    d = float(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64)).max())
    if d > 1e-6:
        ok = False
        print(f"  MISMATCH flatfield: max abs difference {d:.3e}")
    elif verbose:
        print(f"  flatfield agrees to {d:.1e}")
    return ok


if __name__ == "__main__":
    print("vendored-vs-upstream selftest:")
    r = selftest()
    print({True: "  identical", False: "  STALE -- re-copy from upstream",
           None: "  skipped"}[r])
