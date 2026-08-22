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

## What the measurements say, up front

The run is finished (5000 iterations) and every configuration below was measured.

- **The translator works as a translator.** Geometry is preserved: across 68
  hand-marked crack regions, the correlation between a region's local contrast
  before and after translation has a median of **0.921** at the final checkpoint
  (0.982 at iteration 1000). And appearance improves with training -- a classifier
  separates translated patches from real TXM at AUC 0.967, down from 0.994, with
  the output's contrast moving from IQR 0.131 to 0.174 against real TXM's 0.192.
- **The label-transfer claim does not hold.** Four configurations were measured.
  The decisive comparison -- translated SEM against *raw* SEM, which differ in
  nothing but the translator -- comes out **−0.0159, +0.0064, −0.0138, +0.0114**.
  It flips sign, and only one of the four is consistent across seeds. On this
  evidence the answer is **no**, not "not yet".
- **And a trade-off worth knowing about.** The checkpoint with the *best*
  appearance produced the *worst* transfer. Longer training made the outputs more
  TXM-like and simultaneously less useful downstream.

So: the machinery is sound and the geometry property holds, but the thing your PI
asked about -- using SEM images to say something about TXM -- is not delivered by
this route on this data. Section 1 of [Results](#results) is the negative result in
full; nothing in this repo should be cited as showing that it works.

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

All numbers from `runs/cut/final.pt`, iteration 5000, unless a row names another
checkpoint. Reproduce with `./run eval && ./run transfer && ./run figures`.

### 1. Does a transferred SEM label teach a TXM crack detector? No.

Four arms, equal positives and negatives, scored on the four dense hand-drawn TXM
frames the translator never saw:

| arm | pixel AUC | IoU* |
|---|---|---|
| A  real TXM only | 0.8805 ±0.0229 | 0.5173 ±0.0352 |
| B  + translated SEM | 0.8791 ±0.0021 | 0.5106 ±0.0094 |
| C  + raw SEM | 0.8677 ±0.0056 | 0.4993 ±0.0075 |
| D  translated SEM only | 0.8446 ±0.0145 | 0.4841 ±0.0189 |

Nothing beats A. But the number that matters is **B − C**: both arms add SEM crack
labels, and they differ *only* in whether those labels came through the translator.
The experiment was run at four points, and arm C acts as its own control -- it never
touches the translator, so it must be identical whenever the frame set is:

| translator | SEM frames | A | B | C | **B − C** | seeds agree? |
|---|---|---|---|---|---|---|
| iter 500 | 18 | 0.8805 | 0.8736 | 0.8895 | **−0.0159** | yes (all negative) |
| iter 1000 | 18 | 0.8805 | 0.8959 | 0.8895 | **+0.0064** | no |
| iter 5000 | 18 | 0.8805 | 0.8758 | 0.8895 | **−0.0138** | no |
| iter 5000 | 39 | 0.8805 | 0.8791 | 0.8677 | **+0.0114** | yes (all positive) |

Read the control columns first: **A is 0.8805 in all four rows** and **C is 0.8895
in all three rows that share a frame set**, to four decimal places. The harness is
deterministic and the frame set is genuinely pinned, so the movement in B is real
movement in B.

And B − C flips sign twice. The two configurations where every seed agrees point in
**opposite directions**. An effect that changes sign when you change the training
length or the number of source frames is not an effect; it is noise the size of the
±0.0229 seed spread the baseline shows on its own.

![Paired change in pixel AUC against arm A, each seed shown as a dot](figures/paired_deltas.png)

Two further readings, both negative:

- **Longer training made transfer worse.** Holding the frame set at 18, arm B went
  0.8736 → 0.8959 → 0.8758 across iterations 500, 1000, 5000. Non-monotonic, so the
  iteration-1000 peak was a lucky point rather than a trend. An earlier version of
  this README extrapolated from that peak and predicted finishing the run would
  improve things. It did not.
- **The one "consistent" positive is an artefact of C getting worse, not B getting
  better.** In the 39-frame row C falls from 0.8895 to 0.8677 while B barely moves
  (0.8758 → 0.8791). Raw SEM degrades faster than translated SEM as you add source
  frames -- which is a statement about raw SEM scaling badly, not about the
  translation being good.

**Arm D**, trained only on translated SEM with zero real TXM crack labels, reaches
0.8446. Above chance, clearly below the real-TXM baseline, and it got *worse* with
more training (0.8562 at iteration 1000). Translated data is not a substitute for
real labels here.

### 2. Appearance improves with training; transfer does not

| | C2ST AUC vs real TXM | translated IQR | crack-contrast r (median) | top separator |
|---|---|---|---|---|
| iter 1000 | 0.9937 | 0.1312 | 0.982 | `p99` (0.824 alone) |
| iter 5000 | **0.9673** | **0.1744** | 0.921 | `p99` (0.650 alone) |
| real TXM | — | 0.1922 | — | — |

Training clearly helped the *image* problem. The contrast over-compression at
iteration 1000 (IQR 0.131 against a target of 0.192) is largely gone by 5000
(0.174), and the intensity tail that gave the output away weakens from 0.824 to
0.650 as a single-descriptor separator.

Both directions at once, and they oppose: **appearance got better, geometry
retention got slightly worse (r 0.982 → 0.921), and downstream transfer got worse
(B 0.8959 → 0.8758).** That is the useful finding in this section. A generator
pushed harder to satisfy the critic spends some of the fidelity that label transfer
depends on. Anyone extending this should treat "make it look more like TXM" and
"make its labels more transferable" as competing objectives, not the same one.

At AUC 0.967 the outputs are still trivially distinguishable from real TXM. This is
not a solved translation.

![Radially averaged power spectrum: real TXM holds more relative power in the fine bands than either SEM or the translated output](figures/power_spectrum.png)

The spectrum corrects an assumption that was asserted twice in earlier versions of
this README before being measured. TXM looks smoother than SEM, so the expectation
was that it carries less high-frequency content and a good translation should blur.
The opposite holds: in the four finest bands real TXM keeps 0.0188 / 0.0053 /
0.0024 / 0.0019 of its power against SEM's 0.0155 / 0.0042 / 0.0014 / 0.0005. TXM
is *noisier* at fine scale -- photon noise. Matching TXM therefore means *adding*
fine-scale noise, which is also why the `edge_corr` diagnostic falls during
training while coarse structure is untouched: the added noise is uncorrelated with
the input's own fine detail by construction.

### 3. Geometry: preserved, and this part is solid

![SEM input beside its translation: crack outline, grain network and pores in the same places, no visible tile seams](figures/full_frame.png)

Across **68 hand-marked crack regions in 8 frames**, each region's contrast against
a ring of its own local background, before and after translation: per-frame
correlation **median 0.921** at iteration 5000. Contrast is rescaled close to
linearly rather than scrambled, so a mask drawn on the SEM does still describe a
real feature in the output. Combined with the 39-of-39 mask-geometry check in
`./run prep`, the *mechanism* of label transfer is sound -- it is the downstream
benefit that is absent, not the registration.

The honest caveat: only **39%** of marked regions stay *darker* than their
surroundings. Partly the translator flattens the largest features -- a solid-black
through-crack has no counterpart anywhere in flat-fielded TXM, whose IQR is 0.022,
so the critic pushes it toward mid-grey. Partly "darker" is the wrong test:
several frames have marked regions that are *brighter* than their background in the
SEM to begin with (+0.169 on one), because SEM gives a crack bright topographic
edges as well as a dark interior. Proportional retention is the meaningful number;
polarity is not.

![Per-frame crack contrast before and after translation, clustered near the identity line](figures/crack_contrast.png)

### 4. Training

![Training losses on a log scale: critic and generator in a stable adversarial band, both NCE terms declining](figures/training_losses.png)

5.49 M parameters, batch 8, 256 px patches, 5000 iterations. 1.51 s/iter on an idle
M4 Max; this run averaged worse because the host was shared. Both contrastive terms
fall steadily and the adversarial pair stays in a stable band -- no mode collapse,
no critic runaway. The training dynamics are not the problem.

## Limits

- **It does not see under the surface.** The single most likely misreading. This
  renders a SEM frame in the TXM domain; it does not recover subsurface structure,
  and a feature it draws is not evidence of anything at depth.
- **The headline claim is negative, and a negative here is weaker than a positive
  would be.** B − C flips sign across configurations, which rules out a *reliable*
  benefit at this effect size. It does not prove no benefit exists -- a larger
  effect, or a better translator, or more test specimens could still find one. What
  it does rule out is citing this repo as evidence that the method works.
- **The experiment cannot resolve small effects.** The baseline's own seed-to-seed
  spread is ±0.0229 AUC with 3 seeds and 4 test frames. Every A/B/C difference
  measured is at or inside that. Anyone repeating this should budget for far more
  seeds and, more importantly, more test specimens.
- **The four TXM test frames all come from the b2 specimen.** They are the only
  dense hand-drawn ground truth in either repo, so cross-specimen generalisation is
  completely untested, and one specimen's idiosyncrasies could dominate everything
  above.
- **The real-TXM baseline is rule-taught, the SEM arms are human-taught.** Arm A's
  positives come from `write_positive_crack_labels.py`, not an annotator. That
  asymmetry favours the SEM arms, so the negative result is if anything understated.
- **The SEM negatives are mostly inferred, not marked.** 35,020 pixels were
  explicitly hand-marked not-crack; 202.6 M qualified by being more than 50 px from
  anything the reviewer touched. In this dataset an unmarked pixel means "the
  model's opinion stands", not "not a crack", so distance is deciding most of what
  a negative is.
- **IoU* is tuned on the test frames.** An optimistic ceiling, applied equally to
  every arm. AUC is the number to compare.
- **The scale ratio between modalities is unknown.** SEM is recoverable
  (0.042-0.096 um/px from the burned-in bar); TXM is not, so nothing here is
  resampled to a matched physical scale and translation is at 1:1 pixels. If the
  true ratio is far from 1, that alone could explain the negative result -- this is
  the most likely single flaw in the whole setup.
- **62 SEM frames and 66 usable TXM mosaics** -- 2.1 gigapixels and 38k patches,
  but only 9 SEM and 5 TXM independent specimen groups. Pixels are plentiful;
  specimens are not.
- **One architecture, one objective, one hyperparameter setting.** No sweep over
  `lambda_nce`, patch size, or generator depth. The CUT defaults were used
  throughout.

## Licence

Code MIT ([LICENSE](LICENSE)). The micrographs are not redistributed here; they
belong to the two sibling repositories and carry their CC BY 4.0 data licence.
