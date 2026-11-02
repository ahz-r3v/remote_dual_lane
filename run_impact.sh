#!/bin/bash
# Remote-KV slow lane: impact experiment on one node with 2 NVLink-connected GPUs (H200 in the reference run).
# Same protocol as reference_results/run_impact_112_20_original.pbs (ASPIRE 2B job 152114, 2026-10-17), without PBS.
#   G1 = compute GPU: one SGLang process, fast lane = 112 SMs (local KV), slow lane = remaining 20 SMs (green contexts);
#        with ASH_REMOTE_KV=1 the slow lane's full-attention KV pool lives in G0's HBM and FA3 reads/writes it over NVLink.
#   G0 = KV host: also runs its own plain decode ("owner") so its slowdown from serving the remote KV can be measured.
# Shapes: owner 20 x 70k, fast 32 x 50k, slow 16 x 32k; 1500 decode steps per lane (owner 3500).
# Usage: source env.sh; MODEL=/path/to/Qwen3.6-35B-A3B-FP8 GPUS=0,1 OUT=./runs/<tag> bash run_impact.sh [step ...]
#   steps (default all): check v2 full owner fast slow_remote both_remote both_local
#   NSYS=/path/to/nsys enables nsys traces (needed only for per-kernel analysis; step medians come from steps_*.txt).
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); B=$H/bench; S=${SGLANG_ASH:-$H/sglang_ash_r001}
: "${MODEL:?set MODEL}"; : "${PYTHON:?source env.sh first}"
R=${OUT:-$H/runs/$(date +%Y%m%d_%H%M%S)}; mkdir -p $R/tests
VIS=${GPUS:-0,1}; G0=${VIS%%,*}; G1=${VIS##*,}
STEPS_TO_RUN=${*:-check v2 full owner fast slow_remote both_remote both_local}
FAST_B=32; FAST_CTX=51200; SLOW_B=16; SLOW_CTX=32768; OWN_B=20; OWN_CTX=71680; STEPS=1500; OWN_STEPS=3500
FAST_TOK=$(( FAST_B * (FAST_CTX + STEPS + 128) ))
export ASH_FAST_B=$FAST_B ASH_FAST_CTX=$FAST_CTX ASH_SLOW_B=$SLOW_B ASH_SLOW_CTX=$SLOW_CTX ASH_FAST_STEPS=$STEPS ASH_SLOW_STEPS=$STEPS
if [ -n "${NSYS:-}" ]; then PROF() { $NSYS profile -t cuda --cuda-graph-trace=node -f true -o "$1" "${@:2}"; }; else PROF() { "${@:2}"; }; fi
P() { echo "$(date +%T) $*" | tee -a $R/progress.log; }
{ date; hostname; echo "GPUS=$VIS G0=$G0 (KV host) G1=$G1 (compute)"; nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv,noheader
  nvidia-smi topo -m; $PYTHON -c "import torch,sgl_kernel;print('torch',torch.__version__,'sgl_kernel',sgl_kernel.__version__)"
  sha256sum $B/*.py $B/*.yaml $0; } > $R/env.txt 2>&1
LANE_BASE="--model-path $MODEL --tp-size 1 --kv-cache-dtype auto --attention-backend fa3 --disable-radix-cache --mem-fraction-static 0.75 --max-total-tokens $FAST_TOK --max-running-requests $FAST_B --cuda-graph-bs 1 --enable-decode-lane-mux"
OWN="--model-path $MODEL --tp-size 1 --kv-cache-dtype auto --attention-backend fa3 --disable-radix-cache --mem-fraction-static 0.7 --max-running-requests $OWN_B --cuda-graph-max-bs $OWN_B"
wait_file() { for t in $(seq 1 900); do [ -f $1 ] && return 0; kill -0 $2 2>/dev/null || return 1; sleep 1; done; return 1; }

run_mode() {  # $1 mode, $2 lanes yaml, $3 dir tag
  local MODE=$1 Y=$2 D=$R/$3; mkdir -p $D
  local m rem
  case $MODE in solo_fast) m=fast; rem=0 ;; solo_slow_local) m=slow; rem=0 ;; solo_slow_remote) m=slow; rem=1 ;; both_local) m=both; rem=0 ;; both_remote) m=both; rem=1 ;; esac
  local PO=""
  if [ $MODE = both_remote ]; then   # owner decodes on the KV host while the slow lane reads its KV
    ASH_PREFILL_TOKENS=8192 CUDA_VISIBLE_DEVICES=$G0 ASH_ROLE=owner ASH_REMOTE_KV=0 ASH_STEP_LOG=$D/steps_owner.txt ASH_SIGNAL=$D/sig_owner \
      ASH_READY=$D/ready_owner ASH_GO=$D/GO_owner ASH_DREADY=$D/dready_owner ASH_DGO=$D/DGO \
      PROF $D/owner $PYTHON $B/dual_bench.py $OWN --batch-size $OWN_B --input-len $OWN_CTX --output-len $OWN_STEPS --result-filename $D/owner.jsonl < /dev/null > $D/owner.log 2>&1 & PO=$!
    wait_file $D/ready_owner $PO && touch $D/GO_owner && wait_file $D/dready_owner $PO || P "$3 owner failed to get ready"
  fi
  # compute GPU first: inside the process cuda:0 = compute GPU, cuda:1 = KV host
  CUDA_VISIBLE_DEVICES=$G1,$G0 ASH_MODE=$m ASH_REMOTE_KV=$rem ASH_OUT=$D ASH_SIGNAL=$D/sig_lanes ASH_CHECK_SMS=1 \
    PROF $D/lanes $PYTHON $B/dual_lane_bench_r005.py $LANE_BASE --decode-lane-mux-config-path $Y < /dev/null > $D/lanes.log 2>&1 & local PL=$!
  if [ -n "$PO" ]; then wait_file $D/sig_lanes $PL && touch $D/DGO; fi
  wait $PL; local rc=$?
  [ -n "$PO" ] && wait $PO
  P "$3 rc=$rc $(grep -h '"result"\|"sm_check"\|"slow_pools"\|Error\|Traceback' $D/lanes.log | cut -c1-300 | tr '\n' ' ')"
}
owner_alone() {  # $1 dir tag, $2 batch, $3 ctx, $4 steps, $5 GPU
  local D=$R/$1; mkdir -p $D
  ASH_PREFILL_TOKENS=8192 CUDA_VISIBLE_DEVICES=$5 ASH_ROLE=owner ASH_REMOTE_KV=0 ASH_STEP_LOG=$D/steps_owner.txt ASH_SIGNAL=$D/sig_owner \
    PROF $D/owner $PYTHON $B/dual_bench.py --model-path $MODEL --tp-size 1 --kv-cache-dtype auto --attention-backend fa3 --disable-radix-cache \
    --mem-fraction-static 0.7 --max-running-requests $2 --cuda-graph-max-bs $2 --batch-size $2 --input-len $3 --output-len $4 \
    --result-filename $D/owner.jsonl < /dev/null > $D/owner.log 2>&1
  P "$1 steps=$(wc -l < $D/steps_owner.txt 2>/dev/null) $(grep -h 'Traceback\|Error' $D/owner.log | head -2 | tr '\n' ' ')"
}

for s in $STEPS_TO_RUN; do case $s in
  check) P "FA3 lane-split GPU check"
    CUDA_VISIBLE_DEVICES=$G1 $PYTHON $S/sglang/test/ash/test_fa3_lane_split_gpu.py --sms 16,20,32 --batches 16 --ctx 4096,32768 --out $R/tests/fa3_slow.json > $R/tests/fa3_slow.log 2>&1
    P "fa3 slow shapes rc=$? $(tail -1 $R/tests/fa3_slow.log)"
    CUDA_VISIBLE_DEVICES=$G1 $PYTHON $S/sglang/test/ash/test_fa3_lane_split_gpu.py --sms 112 --batches 32 --ctx 51200 --out $R/tests/fa3_fast.json > $R/tests/fa3_fast.log 2>&1
    P "fa3 fast shape rc=$? $(tail -1 $R/tests/fa3_fast.log)" ;;
  v2) run_mode solo_slow_local $B/lanes_112_rest.yaml solo_slow_local ;;   # slow lane alone, KV local (control)
  full) owner_alone full132 $FAST_B $FAST_CTX $STEPS $G1 ;;               # fast load on the whole GPU (baseline)
  owner) owner_alone solo_owner $OWN_B $OWN_CTX $OWN_STEPS $G0 ;;
  fast) run_mode solo_fast $B/lanes_112_rest.yaml solo_fast ;;
  slow_remote) run_mode solo_slow_remote $B/lanes_112_rest.yaml solo_slow_remote ;;
  both_remote) run_mode both_remote $B/lanes_112_rest.yaml both_remote ;;
  both_local) run_mode both_local $B/lanes_112_rest.yaml both_local ;;
  *) P "unknown step $s" ;;
esac; done
P "done $R"; $PYTHON $H/summarize.py $R | tee $R/summary.txt
