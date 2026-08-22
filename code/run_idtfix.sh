#!/usr/bin/env bash
# Controlled retrain with a pixel-space identity loss, then re-measure everything.
# Identical to the baseline run except for --lambda-idt-l1 / --lambda-idt-hf, so the
# comparison against runs/cut/final.pt isolates those two terms. Results go to a
# separate namespace so the baseline numbers stay intact.
set -uo pipefail
cd "$(dirname "$0")"
P=/Users/jiamingzhang/Desktop/sem-crack-detector/.venv/bin/python3
LOG=../out/idtfix.log
R=../runs/cut_idtfix
echo "$(date)  training with pixel identity loss (lambda 5 / 5)" > $LOG

caffeinate -dims $P -u train.py --iters 5000 --batch 8 --log-every 50 \
  --sample-every 1000 --save-every 500 \
  --lambda-idt-l1 5.0 --lambda-idt-hf 5.0 --out $R >> $LOG 2>&1
echo "$(date)  training done: $(tail -1 $R/log.csv)" >> $LOG

cp $R/ckpt.pt $R/final.pt

# 1. the measurement this whole change targets
$P -u eval_identity.py --ckpt $R/final.pt --out ../out/idtfix_identity.json >> $LOG 2>&1
echo "$(date)  identity eval done" >> $LOG

# 2. did fixing fidelity cost appearance, or the translation itself?
$P -u translate.py --ckpt $R/final.pt --masked-only --force --png >> $LOG 2>&1
$P -u eval_translation.py --ckpt $R/final.pt --n-patches 900 \
   --out ../out/idtfix_translation.json >> $LOG 2>&1
echo "$(date)  translation eval done" >> $LOG

# 3. and the question that actually matters
$P -u eval_label_transfer.py --arms A,B,C,D --seeds 3 --rows 60000 \
   --out ../out/idtfix_label_transfer.json >> $LOG 2>&1
$P report_transfer.py --json ../out/idtfix_label_transfer.json >> $LOG 2>&1
echo "$(date)  IDTFIX COMPLETE" >> $LOG
