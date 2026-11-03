#!/bin/bash
# Atomic driver for attn_kernel_cmp.py: one process per (kernel, SMs) unit, gpuguard yield point between units.
# Usage (inside gpuguard, env.sh sourced): OUT=runs/attn_kernel_cmp [UNITS="fa3:20 ..."] bash experiments/tools/attn_kernel_cmp.sh
set -uo pipefail
H=$(cd "$(dirname "$0")/../.." && pwd); R=${OUT:?}; mkdir -p $R; GG=${GPUGUARD_PY:-/home/chengzhi/github/gpuguard}
ALL=""; for k in fa3 fi_decode_cuda fi_decode_tc fi_prefill_fa2 fi_prefill_fa3; do for n in 8 16 20 24; do ALL="$ALL $k:$n"; done; done
for u in ${UNITS:-$ALL}; do
  if [ -n "${GPUGUARD_CTL:-}" ]; then
    PYTHONPATH=$GG python3 -c 'import gpuguard; y = gpuguard.should_yield(); y and gpuguard.yield_gpu(); print(int(bool(y)))' \
      | xargs -I{} echo "$(date +%T) before $u yielded={}" >> $R/yields.log
  fi
  IFS=: read k n ps <<< "$u"; ps=${ps:-16}; lg=$R/${k}_sm${n}_ps$ps.log   # unit = kernel:sms[:page_size]
  CUDA_VISIBLE_DEVICES=${GPU_COMPUTE:-1},${GPU_HOST:-0} ASH_SGLANG=$H/sglang_ash_r001/sglang $PYTHON $H/experiments/tools/attn_kernel_cmp.py $k $n $ps \
    > $lg 2>&1; rc=$?
  grep -h '^{' $lg >> $R/results.jsonl
  echo "$(date +%T) $u rc=$rc $(grep -h 'Traceback\|Error' $lg | tail -1 | cut -c1-200)" >> $R/progress.log
done
