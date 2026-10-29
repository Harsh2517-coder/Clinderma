#!/bin/zsh
# restarts training (resuming from the last checkpoint) until it finishes cleanly, at most 6 attempts
cd "$(dirname "$0")"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.7
for i in 1 2 3 4 5 6; do
  echo "== attempt $i $(date)" >> train.log
  caffeinate -i ./venv/bin/python train.py 20 output >> train.log 2>&1 && { echo "== FINISHED $(date)" >> train.log; exit 0; }
done
echo "== GAVE UP $(date)" >> train.log
