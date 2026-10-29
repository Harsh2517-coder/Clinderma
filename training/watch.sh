#!/bin/zsh
# live training status, refreshes every 20 s. Stop with Ctrl-C (training keeps running).
cd "$(dirname "$0")"
TOTAL=20
while true; do
  clear
  echo "RF-DETR small acne training    $(date '+%a %d %b %H:%M:%S')"
  echo "------------------------------------------------------------"
  if pgrep -f "train.py $TOTAL" >/dev/null; then echo "process:   RUNNING"; else
    if grep -q "== FINISHED" train.log; then echo "process:   FINISHED"; else echo "process:   NOT RUNNING (crashed or stopped, see train.log)"; fi; fi
  echo "attempts:  $(grep -c '^== attempt' train.log) (more than 1 means it crashed and resumed)"
  line=$(tr '\r' '\n' < train.log | grep -E "Epoch [0-9]+: +[0-9]+%" | tail -1)
  ep=$(echo "$line" | sed -E 's/^Epoch ([0-9]+):.*/\1/'); pct=$(echo "$line" | sed -E 's/^Epoch [0-9]+: +([0-9]+)%.*/\1/')
  eta=$(echo "$line" | sed -E 's/.*<([0-9:]+),.*/\1/'); spd=$(echo "$line" | sed -E 's/.*, +([0-9.]+it\/s).*/\1/')
  if [ -n "$ep" ]; then
    done_ep=$((ep))
    echo "progress:  epoch $((done_ep + 1)) of $TOTAL, ${pct}% through it   (${spd})"
    left=$(( (TOTAL - done_ep - 1) * 480 / 60 ))
    echo "ETA:       this epoch ~${eta} left; all epochs ~$((left / 60)) h $((left % 60)) min after that (at ~8 min/epoch)"
  else echo "progress:  starting up..."; fi
  echo
  echo "validation (213 held-out photos), per epoch:"
  [ -f output/metrics.csv ] && ./venv/bin/python - <<'PY'
import csv
rows = [r for r in csv.DictReader(open("output/metrics.csv")) if r.get("val/mAP_50")]
print("  epoch   mAP@50   mAP@50-95   precision   recall   (EMA mAP@50)")
for r in rows[-12:]:
    f = lambda k: f"{float(r[k]):.3f}" if r.get(k) not in (None, "") else "  -  "
    print(f"  {int(float(r['epoch']))+1:>5}   {f('val/mAP_50'):>6}   {f('val/mAP_50_95'):>9}   {f('val/precision'):>9}   {f('val/recall'):>6}   ({f('val/ema_mAP_50')})")
if not rows: print("  (no validation yet)")
PY
  echo
  echo "swap: $(sysctl -n vm.swapusage | sed -E 's/ +\(encrypted\)//')"
  echo "last error in log: $(tr '\r' '\n' < train.log | grep -E 'RuntimeError|Traceback' | tail -1 | cut -c1-120)"
  echo; echo "(Ctrl-C closes this view only)"
  sleep 20
done
