"""Paired reading of the label-transfer arms.

Comparing arm means with their own standard deviations wastes most of the power
here: every arm within a seed is trained on the same row budget and scored on the
SAME cached test set, so the seed-to-seed swing (which is large -- the four test
frames come from one specimen) is shared and cancels in a difference. The paired
delta against arm A is therefore the statistic to read, not the four means.

Reads out/label_transfer.json; no refitting.
"""
import argparse
import json

import numpy as np

import config as C

BASE = "A_real_txm_only"
LABEL = {
    "A_real_txm_only": "A  real TXM only",
    "B_txm_plus_translated_sem": "B  + translated SEM",
    "C_txm_plus_raw_sem": "C  + raw SEM",
    "D_translated_sem_only": "D  translated SEM only",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=str(C.OUT / "label_transfer.json"))
    args = ap.parse_args()
    d = json.loads(open(args.json).read())
    arms = {k: v for k, v in d["arms"].items() if isinstance(v, dict) and "runs" in v}
    if BASE not in arms:
        print(f"no {BASE} in {args.json}")
        return

    seeds = [r["seed"] for r in arms[BASE]["runs"]]
    base = {r["seed"]: r for r in arms[BASE]["runs"]}

    print(f"{len(seeds)} seeds, row budget {d['row_budget']}, "
          f"equalised to {d['notes'].get('equalisation', {}).get('equalised_to')}\n")
    print(f"{'arm':24s} {'AUC':>16s} {'paired delta vs A':>22s} {'IoU*':>16s}")
    for k, v in arms.items():
        runs = {r["seed"]: r for r in v["runs"]}
        auc = np.array([runs[s]["mean_auc"] for s in seeds])
        iou = np.array([runs[s]["mean_iou"] for s in seeds])
        if k == BASE:
            print(f"{LABEL.get(k,k):24s} {auc.mean():8.4f} +-{auc.std():.4f} "
                  f"{'--':>22s} {iou.mean():8.4f} +-{iou.std():.4f}")
            continue
        dl = np.array([runs[s]["mean_auc"] - base[s]["mean_auc"] for s in seeds])
        # sign agreement across seeds is the honest small-n significance statement
        same = "all seeds agree" if (dl > 0).all() or (dl < 0).all() else "signs disagree"
        print(f"{LABEL.get(k,k):24s} {auc.mean():8.4f} +-{auc.std():.4f} "
              f"{dl.mean():+9.4f} +-{dl.std():.4f} {iou.mean():8.4f} +-{iou.std():.4f}")
        print(f"{'':24s} {'':16s}   per seed {np.round(dl,4).tolist()}  {same}")

    # The decisive comparison is not either arm against A but B against C: both add
    # SEM crack labels, and they differ ONLY in whether those labels arrived through
    # the translator. If B does not beat C, the translation is not what helped.
    bk, ck = "B_txm_plus_translated_sem", "C_txm_plus_raw_sem"
    if bk in arms and ck in arms:
        br = {r["seed"]: r["mean_auc"] for r in arms[bk]["runs"]}
        cr = {r["seed"]: r["mean_auc"] for r in arms[ck]["runs"]}
        dl = np.array([br[s] - cr[s] for s in seeds])
        verdict = ("consistent" if (dl > 0).all() or (dl < 0).all()
                   else "NOT RESOLVED -- signs disagree across seeds")
        print(f"\nB minus C (translated vs raw SEM, the control that matters):")
        print(f"  {dl.mean():+.4f} +-{dl.std():.4f}   per seed "
              f"{np.round(dl,4).tolist()}   {verdict}")

    a = np.array([base[s]["mean_auc"] for s in seeds])
    print(f"\nSeed-to-seed spread of the baseline alone: +-{a.std():.4f} AUC.")
    print("An effect smaller than roughly twice that is not resolvable with "
          f"{len(seeds)} seeds\nand 4 test frames from a single specimen -- which "
          "is the situation for A vs B vs C.")


if __name__ == "__main__":
    main()
