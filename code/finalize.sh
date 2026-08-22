#!/usr/bin/env bash
# Wait for training to finish, then redo every measurement against the final
# checkpoint, this time over ALL 39 masked frames rather than the 18 the
# contended first pass settled for. Nothing here is interactive; it is the "finish the run and
# re-measure" step the README names as the most useful next thing.
set -uo pipefail
cd "$(dirname "$0")"
P=/Users/jiamingzhang/Desktop/sem-crack-detector/.venv/bin/python3
LOG=../out/finalize.log
echo "$(date)  waiting for training to finish" > $LOG
while pgrep -f "train.py --iters" > /dev/null; do sleep 60; done
echo "$(date)  training done: $(tail -1 ../runs/cut/log.csv)" >> $LOG

cp ../runs/cut/ckpt.pt ../runs/cut/final.pt
echo "$(date)  froze final.pt" >> $LOG

$P -u translate.py --ckpt ../runs/cut/final.pt --masked-only --force --png >> $LOG 2>&1
echo "$(date)  retranslated" >> $LOG

$P -u eval_translation.py --ckpt ../runs/cut/final.pt --n-patches 900 >> $LOG 2>&1
echo "$(date)  eval_translation done" >> $LOG

$P -u eval_label_transfer.py --arms A,B,C,D --seeds 3 --rows 60000 \
    --out ../out/label_transfer.json >> $LOG 2>&1
echo "$(date)  transfer done" >> $LOG

$P report_transfer.py >> $LOG 2>&1
$P make_figures.py >> $LOG 2>&1
$P make_panels.py --ckpt ../runs/cut/final.pt >> $LOG 2>&1
echo "$(date)  FINALIZE COMPLETE" >> $LOG
