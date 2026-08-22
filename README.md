# sem2txm

Translate a SEM micrograph into the TXM domain, and test whether that lets a
hand-drawn SEM crack label teach a TXM crack detector.

Companion to [sem-crack-detector](https://github.com/jzhang29-max/sem-crack-detector)
and [TXM_Crack_Detection_Pipeline](https://github.com/jzhang29-max/TXM_Crack_Detection_Pipeline).
It reads its data from both rather than shipping a third copy of the micrographs.

> **Read this first.** A SEM image is of a surface. A TXM image is a line integral
> through the specimen. Nothing in a surface image determines what lies beneath it,
> so this tool **does not predict subsurface structure** and a feature it draws is
> not evidence that the feature exists at depth. What it does is re-render a SEM
> frame in the TXM domain with its geometry held in place. Whether that is *useful*
> is a separate, measurable question, and it is the one this repo actually answers.

## What is being claimed

Two things, in order of how well they are supported:

1. **Label transfer.** TXM crack labels are scarce here -- four hand-drawn frames.
   SEM crack labels are not -- 39 hand-drawn masks. If a SEM frame can be moved
   into the TXM domain without moving its cracks, its mask comes along for free.
   Measured against three controls in [Results](#results).
2. **Visualisation.** Given a SEM frame, show what that field would plausibly look
   like under TXM. This is the thing that is easy to look at and hard to verify;
   it is reported second on purpose.

## Install and run

```bash
git clone https://github.com/jzhang29-max/sem2txm.git
cd sem2txm && ./run
```

`./run` builds its own virtualenv on first use and then executes every stage in
order. Individual stages:

```bash
./run prep        # put both modalities in one domain, cut patch banks
./run train       # train the translator
./run eval        # appearance + geometry measurements
./run transfer    # the label-transfer experiment
./run figures     # regenerate every figure below
./run scale       # recover SEM um/px from the burned-in scale bar
./run groups      # print the specimen grouping used for splits
```

Both sibling repos are expected next to this one; override with
`SEM2TXM_SEM_REPO` / `SEM2TXM_TXM_REPO`. Every stage is resumable and skips work
already done. Measured on an M4 Max (36 GB, torch on MPS): `prep` about 35
minutes for 2.1 gigapixels, `train` 1.51 s/iter at batch 8.

## Method

**Unpaired, because the data is unpaired.** SEM `316_h_b2` and TXM `260618_b2` are
the same specimen -- TXM first (18-20 June), SEM after (22 June, 8 July), the
order an in-situ load sequence followed by post-mortem SEM gives you. But no frame
is registered to any mosaic, and nothing in either repo records a transform. There
is no per-pixel target to regress against.

**So the content constraint is contrastive, not a reconstruction loss.** A plain
GAN only has to look like TXM, not like *this input* -- it is free to move a crack,
and a crack that has drifted takes its mask with it. The
[CUT](https://arxiv.org/abs/2007.15651) objective instead requires a patch of the
output to be more similar to the *same* patch of the input than to any other patch
of it. That pins content in place without ever needing a registered pair, and it
is the reason label transfer is even coherent here.

Three terms:

| term | what it buys |
|---|---|
| adversarial (LSGAN, PatchGAN critic) | output looks like a flat-fielded TXM mosaic |
| PatchNCE | output stays where the input put it |
| identity NCE | `G(real TXM) ~ real TXM`, so the transform is not applied to material already in the target domain |

**The generator bottleneck is windowed self-attention** (shifted between blocks, as
in Swin), not the residual convolutions the original CUT used. Cracks are long,
thin and continuous over hundreds of pixels; stacked 3x3 convolutions reach that
far only through depth, whereas attention inside an 8x8 window at stride 4 relates
points 32 px apart in one hop and the shift carries it across window borders.
5.49 M parameters.

**One domain for both modalities.** A raw TXM mosaic is dominated by the beam and
thickness envelope -- at 512 px a raw crop is mostly a smooth gradient, so a
translator trained raw-to-raw would spend its capacity inventing an envelope:

![Raw TXM crops beside destitched and flat-fielded versions of the same crops; the raw column is smooth gradient, the corrected column shows linear crack-like features and texture](figures/preprocessing.png)

Both modalities are therefore put through the same operator family the TXM app
already uses for its human-facing view -- destitch (TXM only; SEM has no tile grid)
then pseudo-flat-field -- by **importing `destitch.py` and `flatfield.py` from that
repository** rather than reimplementing them. So the domain this tool translates
*into* is exactly the domain that repo's reviewers mark cracks on. Reproduced on
`260618_b2_337_19`: median 1.0007, IQR 0.0220, against the 0.019-0.036 that repo
documents.

Full data provenance, and what it does and does not support: [docs/DATA.md](docs/DATA.md).

## Two things that had to be fixed first

**The SEM info panel.** About 280 rows at the bottom of each capture are an
instrument panel -- date, kV, HFW, detector, scale bar -- not material. Trained on,
a translator learns to generate text. Cropped using `find_field_of_view` from the
SEM repo, which was validated there across all 62 frames.

The check that licenses label transfer: the 39 correction masks were painted on the
**cropped** frame, so a correct crop reproduces their shape exactly. It does, for
**39 of 39 with zero mismatches**, reported by `./run prep` on every run. The mask
registers pixel-for-pixel, and flat-fielding preserves geometry, so it still
registers on the translated output.

**The pixel scale, which turned out to be recoverable.** The shipped TIFFs carry no
FEI/ZEISS pixel size and the SEM tool's provenance says so plainly
(`"calibrated": false, "um_per_px": null`). But the instrument burned it into the
panel. `./run scale` measures the drawn bar -- whose label sits *inside* it, so the
longest run of light pixels is only half the bar:

```
260622_316_H_b2_front_CBS_01:  bar span 2372 px
  100 um / 2372 px = 0.0422 um/px  ->  HFW = 259 um
```

259 um is the HFW printed on that same panel, and a 3072-px-wide frame gives the
same 259 um independently. Across the 51 frames that carry a panel: 7
magnifications, **0.042 to 0.096 um/px** (`out/sem_scale.json`).

This does **not** close the scale question, for two reasons. The conversion assumes
every bar reads 100 um -- confirmed for the 259 um setting, not the others. And no
equivalent recovery exists for TXM: the `.xrm` metadata did not survive conversion
either and those mosaics carry no panel. So the SEM-to-TXM pixel *ratio* is still
unknown, training happens in pixel space, and `--sem-um-per-px` /
`--txm-um-per-px` are left unset. Supply both and the ratio is applied and
reported; supply one and it is refused, the convention `semcrack.py` already uses.

## Results

Measured, not yet written -- `./run train` is in progress. This section will carry
the four-arm table, the two-sample test, the crack-contrast retention figure and
the training curves.

## Limits

- **It does not see under the surface.** Stated at the top and repeated here
  because it is the single most likely misreading.
- **The scale ratio between modalities is unknown**, so translation is at 1:1
  pixels and the two domains' effective resolutions differ. The power spectrum
  figure shows by how much.
- **The four TXM test frames are all from the b2 specimen.** They are the only
  dense hand-drawn ground truth that exists here, so cross-specimen
  generalisation of the transfer result is untested.
- **The real-TXM baseline is rule-taught.** Its positives come from
  `write_positive_crack_labels.py`, not an annotator, while the SEM positives are
  hand-drawn. That asymmetry favours the SEM arms.
- **62 SEM frames and 66 usable TXM mosaics.** Large in pixels (2.1 gigapixels,
  38k patches) but small in independent specimens -- 9 SEM groups, 5 TXM groups.

## Licence

Code MIT ([LICENSE](LICENSE)). The micrographs are not redistributed here; they
belong to the two sibling repositories and carry their CC BY 4.0 data licence.
