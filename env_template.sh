# Environment for run_impact.sh. Copy to env.sh, adapt the paths, then `source env.sh`.
# Reference run (ASPIRE 2B, 2 x H200 NVLink, 2026-10-17): Python 3.12, CUDA 13.0, torch 2.11.0+cu130, triton 3.6.0,
# sglang-kernel (sgl_kernel) 0.4.3 (provides sgl_kernel.spatial green-context streams and FA3), flashinfer-python 0.6.12,
# sgl-deep-gemm 0.1.2, transformers 5.8.1, numpy 2.3.5, xgrammar 0.2.1.
# The SGLang Python package must come from ./sglang_ash_r001 (NOT a pip-installed sglang): it is put first on PYTHONPATH.
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHON=${PYTHON:-python3.12}                 # interpreter of a venv with the packages above
export PYTHONPATH=$H/sglang_ash_r001/sglang/python:$H/sglang_ash_r001/sglang:$H/sglang_ash_r001:${PYTHONPATH:-}
C=${TMPDIR:-/tmp}/${USER}_remote_kv_slowlane
export XDG_CACHE_HOME=$C/xdg TRITON_CACHE_DIR=$C/triton TORCH_EXTENSIONS_DIR=$C/torch_ext FLASHINFER_WORKSPACE_BASE=$C/fi \
       TVM_FFI_CACHE_DIR=$C/tvm TORCHINDUCTOR_CACHE_DIR=$C/inductor
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 SGLANG_MOE_EXPERT_READ_TRACKING_MODE=off
# export NSYS=/path/to/nsys     # optional: per-kernel traces
