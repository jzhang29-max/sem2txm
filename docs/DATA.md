# The data, and what it does and does not support

Everything here was checked against the files rather than assumed, because the
central question -- can a SEM frame say anything about a TXM frame -- turns
entirely on how the two sets relate.

## The two sets do cover the same specimens

Grouping both sets by specimen (`./run groups`) lines up like this:

| SEM group | frames | TXM group | mosaics |
|---|---|---|---|
| `316_h_b2` | 21 | `260618_b2` | 17 |
| `316_amb_b3` | 2 | `260618_b3` + `260620_b3` | 13 |
| `316_h_b4` | 2 | — | — |
| `mar_amb_as` / `_cast` / `_hip` | 11 / 10 / 13 | — | — |
| `as_24hr` / `cast_24hr` / `hip_24hr` | 1 / 1 / 1 | — | — |
| — | — | `260619_hc_316l` | 27 |
| — | — | `260620_wrought_316l` | 14 |

So `b2` and `b3` were imaged in both modalities, TXM first (18-20 June) and SEM
after (22 June, 8 July) -- the order you get from an in-situ TXM load/fatigue
sequence followed by post-mortem SEM. That matters: the two sets are two views of
the same material and the same damage, not two unrelated datasets.

## But no pair of frames is registered

Same specimen is not the same field of view. Nothing in either repository records
a transform between a SEM frame and a TXM mosaic, the fields of view differ, and
the modalities see different things by construction -- SEM the surface, TXM the
line integral through the specimen. There is no pixel correspondence to regress
against, which is why the method here is unpaired and why the content term is
contrastive rather than an L1 to a target.

**Consequence to keep in view:** a surface image does not determine subsurface
structure. This tool renders a SEM frame in the TXM domain; it does not see
underneath the surface, and a feature it draws is not evidence that the feature
exists at depth. See the limits section of the README.

## Pixel scale: it was recoverable after all

The shipped TIFFs carry no FEI/ZEISS pixel-size tag -- `tifffile` rewrote them
and `XResolution` is `(1,1)`. The SEM tool's own provenance agrees:

    "calibrated": false, "um_per_px": null,
    "note": "UNCALIBRATED: lengths are in PIXELS."

The instrument had, however, burned the numbers into the bottom of every capture:
date, kV, current, WD, **HFW**, detector, and a drawn scale bar. `./run scale`
crops that panel out and measures the drawn bar.

The bar carries its label *inside* it (`|----100 um----|`), so the longest run of
light pixels is only half the bar. Measuring the span instead, on
`260622_316_H_b2_front_CBS_01`: span 2372 px, and

    100 um / 2372 px = 0.0422 um/px   ->   HFW = 0.0422 x 6144 = 259 um

which is the HFW printed on that same panel. Two independent readings of one
quantity, agreeing. Across the 51 frames that carry a panel there are 7 distinct
magnifications, implying HFW 259 / 276 / 296 / 319 / 346 / 415 um and pixel sizes
of **0.042 to 0.096 um/px**. `out/sem_scale.json` has the per-frame table.

Two cautions, both load-bearing:

- The conversion assumes **every** bar reads 100 um. That is confirmed for the
  259 um setting (by its printed HFW, and independently by a 3072-px-wide frame
  giving the same 259 um) and **not** confirmed for the others. Read their panels
  before quoting them.
- No equivalent recovery exists for TXM: the `.xrm` metadata did not survive the
  conversion to `.tif` either, and those mosaics carry no burned-in panel. So the
  **ratio** between a SEM pixel and a TXM pixel is still unknown, and training
  therefore happens in pixel space with `--sem-um-per-px` / `--txm-um-per-px`
  left unset. Supply both and the ratio is applied and reported; supply one and
  it is refused, following the convention `semcrack.py` already uses.

## One domain for both modalities

A raw TXM mosaic is dominated by the beam and thickness envelope. At 512 px a raw
crop is mostly a smooth gradient, so a translator trained raw-to-raw would spend
its capacity inventing an envelope. Both modalities are therefore put through the
same operator family the TXM app already uses for its human-facing view --
destitch (TXM only; SEM has no tile grid) then pseudo-flat-field -- by importing
`destitch.py` and `flatfield.py` from that repository rather than reimplementing
them. Reproduced on `260618_b2_337_19`: median 1.0007, IQR 0.0220, against the
0.019-0.036 that repo documents.

`figures/preprocessing.png` is the before/after.

## The SEM info panel is cropped, and the crop is checked

Rows below the specimen are an instrument panel, not material -- about 280 rows on
a 4376-row capture. A translator trained on it learns to generate text. The crop
uses `find_field_of_view` from the SEM repo, which was validated there across all
62 frames.

The check that matters: the 39 hand-drawn correction masks were painted on the
**cropped** frame, so a correct crop reproduces their shape exactly. It does, for
**39 of 39, with zero mismatches** (`prep.py` reports this every run). That is what
licenses carrying a SEM mask onto a translated frame: the mask registers
pixel-for-pixel, and flat-fielding preserves geometry.

## Labels, and how much each kind is worth

| source | positives | provenance |
|---|---|---|
| SEM correction masks | 39 frames, e.g. 411,002 px on one frame | **hand-drawn** |
| TXM dense GT | 4 frames, 18.6-29.7% crack | **hand-drawn**; the test set |
| TXM app labels | 30.2 M crack px over 56 images, vs 384.9 M not-crack | **rule-derived** |

The TXM positives outside the four dense frames come from
`write_positive_crack_labels.py` -- dark-relative-to-local, elongated,
inside-specimen, high-confidence cores only. A principled rule, but a rule, not an
annotator. So the real-TXM baseline arm is rule-taught while the SEM arms are
human-taught. That asymmetry favours the SEM arms and is stated in the results
rather than buried.

## Held out, by specimen

`REFERENCE_SPECIMENS = ("333_75", "336_25", "338_13", "343_75")` -- 5 of 71
mosaics -- are excluded from the patch banks in `prep.py`, so the translator never
sees them. Four of them carry dense GT and are the label-transfer test set; the
fifth is a second field of view of `343_75` and would leak into it. This follows
the exclusion the TXM repo already established, and for the same reason: a gate
graded with part of the answer key in the training set means nothing.

## Two sampling bugs the checks caught

Both would have produced plausible-looking numbers, which is why the checks are
asserted in code rather than left to inspection.

**Feature windows need a 192 px margin, not 32.** The per-pixel features come from
`compute_feature_stack`, and the experiment needs them for every labelled pixel of
every image, per seed -- hundreds of whole-frame passes over up to 23 megapixels.
Computing them on windows instead is far cheaper and, for a pixel far enough
inside the window, identical. "Far enough" was first set at 32 px by reading
`GRADIENT_SIGMAS` and `LAPLACIAN_SIGMAS`, which stop at 8. But `SMOOTH_SIGMAS`
runs to **64**, and comparing window features against a whole-frame pass showed
1.8e-2 of error concentrated in `smooth_s16/32/64`. At 192 px the residual is
1.5e-4, in `smooth_s64` alone.

**Random window origins do not sample a frame uniformly.** With a 384 px core, a
pixel five rows from the frame edge lies inside 6 possible windows while a central
pixel lies inside 384 -- so the centre is over-represented about 64-fold. The
cracks in these frames are central, so the test set came out at 42.3 / 35.8 / 39.1
/ 24.9% crack against true whole-frame prevalence of 25.5 / 27.0 / 29.7 / 18.6%.
IoU measured against the wrong prevalence is simply the wrong IoU.

The fix is to tile the frame with non-overlapping cores -- every pixel belongs to
exactly one tile -- and to allocate each tile a share of the sample proportional
to its area, so the smaller edge tiles are neither over- nor under-weighted.
Frames are reflect-padded by the margin first, which is not a convenience:
`compute_feature_stack` filters with scipy's default `mode="reflect"`, so a
reflect-padded window reproduces exactly what a whole-frame pass computes at the
border.

`build_test` now prints sampled prevalence beside whole-frame prevalence for every
test frame and flags any gap over 3 points, so this cannot regress quietly:

```
test 260618_B2_333_75_um_zoom   n=119998 pos=25.6% (whole frame 25.5%)
test 260618_b2_336_25           n=120006 pos=27.2% (whole frame 27.0%)
test 260618_b2_338_13           n=119993 pos=29.8% (whole frame 29.7%)
```

29.8% against the 29.7% the TXM repo documents for that frame is also a check that
this repo is reading the same ground truth that one does.
