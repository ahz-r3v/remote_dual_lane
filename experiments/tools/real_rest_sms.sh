#!/bin/bash
# Real-weight slow lane, local KV, nsys: non-attention ("rest") time per step for {20,12} SMs x {16,8} requests at 32k.
# Usage (inside gpuguard, env.sh sourced): MODEL=... OUT=runs/real_rest_sms bash experiments/tools/real_rest_sms.sh
# Optional: CONFIGS="sms:reqs ..." (default all four), TAG (dir suffix), CTX, STEPS, EXTRA_ARGS, NO_NSYS=1,
#           ASH_NO_CUBLAS_CARVEOUT=1 (passed through; dir suffix _nocarve), REMOTE=1 (slow KV on the peer GPU; dir suffix _remote)
set -uo pipefail
H=$(cd "$(dirname "$0")/../.." && pwd); B=$H/bench; R=${OUT:?}; mkdir -p $R
LANE_BASE="--model-path $MODEL --tp-size 1 --kv-cache-dtype auto --attention-backend fa3 --disable-radix-cache --mem-fraction-static 0.75 --max-total-tokens $((32 * (51200 + 1500 + 128))) --max-running-requests 32 --cuda-graph-bs 1 --enable-decode-lane-mux"
Y20=$B/lanes_112_rest.yaml; Y12=$H/experiments/tools/lanes_120_rest.yaml
CAR=$([ "${ASH_NO_CUBLAS_CARVEOUT:-0}" = 1 ] && echo _nocarve || echo "")$([ "${REMOTE:-0}" = 1 ] && echo _remote || echo "")
GG=${GPUGUARD_PY:-/home/chengzhi/github/gpuguard}
for c in ${CONFIGS:-20:16 12:16 20:8 12:8}; do
  # gpuguard yield point: each config is a separate process, so nothing holds the GPUs here; a waiting job of higher priority
  # runs now and this script resumes once the GPUs are ours again and idle
  if [ -n "${GPUGUARD_CTL:-}" ]; then
    PYTHONPATH=$GG python3 -c 'import gpuguard, sys; y = gpuguard.should_yield(); y and gpuguard.yield_gpu(); print(int(bool(y)))' \
      | xargs -I{} echo "$(date +%T) before $c yielded={}" >> $R/yields.log
  fi
  n=${c%%:*}; b=${c##*:}; case $n in 20) Y=$Y20 ;; 12) Y=$Y12 ;; *) Y=$H/experiments/tools/lanes_104_$n.yaml ;; esac
  set -- $n $b $Y; D=$R/sm$1_b$2$CAR${TAG:-}; mkdir -p $D
  [ -n "${NO_NSYS:-}" ] && PROF="" || PROF="$NSYS profile -t cuda --cuda-graph-trace=node -f true -o $D/lanes"
  CUDA_VISIBLE_DEVICES=${GPU_COMPUTE:-1},${GPU_HOST:-0} ASH_MODE=slow ASH_REMOTE_KV=${REMOTE:-0} ASH_SLOW_B=$2 ASH_SLOW_CTX=${CTX:-32768} ASH_SLOW_STEPS=${STEPS:-300} ASH_NO_CUBLAS_CARVEOUT=${ASH_NO_CUBLAS_CARVEOUT:-0} \
    ASH_OUT=$D ASH_CHECK_SMS=1 ASH_DUMP_LOGITS=0 SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
    $PROF $PYTHON $B/dual_lane_bench_r005.py $LANE_BASE ${EXTRA_ARGS:-} --decode-lane-mux-config-path $3 < /dev/null > $D/lanes.log 2>&1; rc=$?
  echo "$(basename $D) carveout_off=${ASH_NO_CUBLAS_CARVEOUT:-0} rc=$rc $(grep -h '"sm_check"\|"result"\|Traceback\|Error' $D/lanes.log | cut -c1-250 | tr '\n' ' ')" | tee -a $R/progress.log
done
