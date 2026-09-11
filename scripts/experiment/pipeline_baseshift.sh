#!/bin/bash
# The base-shift test. For each offset: the seven can_base2 chunks re-solved from the shifted base
# (shift_plan.py -> export), then one Isaac boot replaying BOTH the base-0 rows (fixed reference:
# misses by the shift) and the re-solved rows (retargeted reference: should not) with the arm
# standing at the shifted base. Results append to $B/results.jsonl; log in $B/pipeline.log.
#   OFFSETS="0 x+2 y-4" DWELL=8 bash pipeline_baseshift.sh
set -o pipefail; cd /e/work/Isaacsim/OmniBase
PY=/e/work/Isaacsim/.lerobot/Scripts/python.exe; B=${B:-/e/data/baseshift}; P=/e/data/fastumi/plans2/can_base2.json
S=/e/work/Isaacsim/OmniBase/scripts/experiment; TASK="pick up the object and put it in the container"
exp() {   # exp <tag> <dx> <dy> <dz>
  [ -d "$B/can_$1" ] && return 0
  "$PY" "$S/shift_plan.py" "$P" $2 $3 $4 "$B/plan_$1.json" | tail -1
  "$PY" -m omnibase export "$B/plan_$1.json" --out "$B/can_$1" --fps 10 --size 512x384 --grip=-1,1 --repeat 1 \
      --task "$TASK" --camera a_wrist_right --min-frames 20 --max-step 60 2>&1 | grep -E "frames in|dropped" | head -3
}
exp 0 0 0 0
for tag in ${OFFSETS:-0 x+2 x-2 y+2 y-2 x+4 x-4 y+4 y-4 x+6 x-6 y+6 y-6 x+8 x-8 y+8 y-8}; do
  if [ "$tag" = 0 ]; then off=0,0,0; roots="$B/can_0"; else
    ax=${tag:0:1}; cm=${tag:1}; m=$(python -c "print($cm/100)")
    case $ax in x) off="$m,0,0";; y) off="0,$m,0";; z) off="0,0,$m";; esac
    exp "$tag" ${off//,/ }; roots="$B/can_0 $B/can_$tag"
  fi
  grep -q "\"tag\": \"$tag\"" "$B/results.jsonl" 2>/dev/null && { echo "skip $tag"; continue; }
  echo "=== $tag  offset $off  $(date +%H:%M:%S)"; rm -f "$B/raw_$tag.jsonl"
  ( cd /e/work/Isaacsim/so101-scene && /e/work/Isaacsim/env/python.exe scripts/grasp_check.py --config configs/eval_can_p1.yaml \
      --root $roots --base-offset "$off" --out "$B/raw_$tag.jsonl" --dwell ${DWELL:-8} 2>&1 | grep -E "^episode|^==|LIFTED|not lifted|jaws commanded|Traceback|Error" )
  [ -f "$B/raw_$tag.jsonl" ] && sed "s/^{/{\"tag\": \"$tag\", /" "$B/raw_$tag.jsonl" >> "$B/results.jsonl" || echo "NO RESULT $tag"
done 2>&1 | tee -a "$B/pipeline.log"
echo "BASESHIFT DONE $(date +%H:%M:%S)" | tee -a "$B/pipeline.log"
