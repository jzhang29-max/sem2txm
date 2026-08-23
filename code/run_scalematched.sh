#!/usr/bin/env bash
# SEM downsampled 2.9x so both modalities sit at the same physical scale for the
# first time -- the ratio measured from the two supplied pairs (2.56 and 3.18).
# Critic input stays high-pass, which was the best texture setting found.
set -uo pipefail
cd "$(dirname "$0")"
P=/Users/jiamingzhang/Desktop/sem-crack-detector/.venv/bin/python3
L=../out/scalematched.log
R=../runs/cut_s29
echo "$(date)  SEM downsample 2.9, highpass critic" > $L
caffeinate -dims $P -u train.py --iters 5000 --batch 8 --log-every 50 \
  --sample-every 1000 --save-every 500 --critic-input highpass \
  --bank-suffix _s29 --out $R >> $L 2>&1
echo "$(date)  training done: $(tail -1 $R/log.csv)" >> $L
cp $R/ckpt.pt $R/final.pt
$P -u eval_identity.py --ckpt $R/final.pt --out ../out/s29_identity.json >> $L 2>&1
$P -u eval_translation.py --ckpt $R/final.pt --n-patches 900 \
   --out ../out/s29_translation.json >> $L 2>&1
# the decisive test: fidelity against the two REAL registered pairs
$P -u run_pairs.py --sem-downsample 2.9 --ratios 1,1.5,2,2.5,3 --angles=-6,-3,0,3,6 \
   --ckpt $R/final.pt --out ../out/pairs_s29 >> $L 2>&1
echo "$(date)  SCALEMATCHED COMPLETE" >> $L
