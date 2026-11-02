# sglang_ash r001 on ASPIRE 2B (H200): interpreter / site-packages of the copied gaas env (env_gaas_q36), SGLang source = this tree.
# Usage inside a PBS job: source <this file> [SRC override for the frozen reference tree]
module load cuda/13.0
ASH_PYENV=/scratch/users/ntu/bzhou011/ash/env_gaas_q36   # (named so it does not clobber callers' variables)
ASH_SGLANG_SRC=${1:-/scratch/users/ntu/bzhou011/ash/sglang_ash/r001}
export LD_LIBRARY_PATH=$ASH_PYENV/syslibs:$ASH_PYENV/python312/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$ASH_SGLANG_SRC/sglang/python:$ASH_SGLANG_SRC/sglang:$ASH_SGLANG_SRC:$ASH_PYENV/mooncake:$ASH_PYENV/venv-sglang/lib/python3.12/site-packages:$ASH_PYENV/pytorch/lib/python3.12/site-packages
export PATH=$ASH_PYENV/venv-sglang/bin:$PATH PYTHON=$ASH_PYENV/python312/bin/python3.12
_ASH_LT=${TMPDIR:-/tmp}
export TORCH_EXTENSIONS_DIR=$_ASH_LT/torch_ext XDG_CACHE_HOME=$_ASH_LT/xdg TRITON_CACHE_DIR=$_ASH_LT/triton FLASHINFER_WORKSPACE_BASE=$_ASH_LT/fi TVM_FFI_CACHE_DIR=$_ASH_LT/tvm TORCHINDUCTOR_CACHE_DIR=$_ASH_LT/inductor
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 SGLANG_MOE_EXPERT_READ_TRACKING_MODE=off
export NSYS_BIN=$(ls /opt/nvidia/hpc_sdk/Linux_x86_64/*/profilers/13.0/Nsight_Systems/bin/nsys 2>/dev/null | tail -1)
