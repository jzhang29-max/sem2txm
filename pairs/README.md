# Registered-pair candidates

Four frames, two pairs, used by `./run pairs` via `../pairs.json`.

| pair | SEM | TXM |
|---|---|---|
| B2 | `260708_316_H_b2_front_CBS_002.tif` | `260618_b2_343_75_LARGE` |
| B3 | `260622_316_amb_b3_CBS_01.tif` | `260620_b3_388_13um_LARGE_2` |

**The images themselves are not committed** (254 MB, and they belong to the two
sibling repositories' datasets under CC BY 4.0). This directory ships with only its
manifest. To reproduce, place the four files here in the layout above.

## Why these SEM files matter beyond the pairing

They are **originals off the instrument, with the `FEI_HELIOS` tag intact**. The
copies in `sem-crack-detector/original/` were rewritten by `tifffile` and lost it,
which is why both sibling repos record the pixel size as unknown
(`"calibrated": false, "um_per_px": null`).

These two carry it exactly:

| frame | `Scan.PixelWidth` | `EBeam.HFW` |
|---|---|---|
| `260708_316_H_b2_front_CBS_002` | 0.103766 um/px | 318.8 um |
| `260622_316_amb_b3_CBS_01` | 0.042155 um/px | 259.0 um |

That is what validated the burned-in-scale-bar recovery in `code/read_scale.py` to
0.07% and 0.01%, at two different magnifications — see the pixel-scale section of the
top-level README. **If these files are ever regenerated or re-exported, preserve the
metadata**; it is the only exact SEM calibration in the project.

## What was measured from them

Both pairs register (same crack, confirmed by checkerboard overlay), and both show
the translated image predicting the real TXM *worse* than the raw SEM does. B3's
block residual is 34.8 px median, which is too coarse for a pixel-aligned paired
loss. Details in README section 2.
