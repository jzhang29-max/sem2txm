#!/usr/bin/env bash
# One change from the original baseline: what the critic is shown. lambda_nce stays
# 1.0, no pixel identity terms -- so any difference is attributable to the critic
# input alone.
set -uo pipefail
cd "$(dirname "$0")"
P=/Users/jiamingzhang/Desktop/sem-crack-detector/.venv/bin/python3
L=../out/highpass.log
R=../runs/cut_hp
echo "$(date)  critic input = highpass (sigma 4), everything else baseline" > $L
caffeinate -dims $P -u train.py --iters 5000 --batch 8 --log-every 50 \
  --sample-every 1000 --save-every 500 --critic-input highpass \
  --out $R >> $L 2>&1
echo "$(date)  training done: $(tail -1 $R/log.csv)" >> $L
cp $R/ckpt.pt $R/final.pt
$P -u eval_identity.py --ckpt $R/final.pt --out ../out/hp_identity.json >> $L 2>&1
$P -u translate.py --ckpt $R/final.pt --masked-only --force --png >> $L 2>&1
$P -u eval_translation.py --ckpt $R/final.pt --n-patches 900 \
   --out ../out/hp_translation.json >> $L 2>&1
$P -u eval_label_transfer.py --arms A,B,C,D --seeds 10 --rows 60000 \
   --out ../out/hp_label_transfer.json >> $L 2>&1
$P report_transfer.py --json ../out/hp_label_transfer.json >> $L 2>&1
$P compare_models.py --ckpts "original=runs/cut/final.pt" \
   "pixel-identity=runs/cut_idtfix/final.pt" "nce0.25=runs/cut_rebal/final.pt" \
   "highpass-critic=runs/cut_hp/final.pt" --at 1536,1280 \
   --out ../figures/four_way_comparison.png >> $L 2>&1
echo "$(date)  HIGHPASS COMPLETE" >> $L
