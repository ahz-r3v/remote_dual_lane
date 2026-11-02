from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _low_resource_linear_kernel(
    x,
    weight,
    bias,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_idxs = k0 + offs_k
        x_vals = tl.load(
            x + offs_m[:, None] * stride_xm + k_idxs[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k_idxs[None, :] < K),
            other=0.0,
        )
        w_vals = tl.load(
            weight + offs_n[None, :] * stride_wn + k_idxs[:, None] * stride_wk,
            mask=(offs_n[None, :] < N) & (k_idxs[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(x_vals, w_vals)

    if HAS_BIAS:
        bias_vals = tl.load(bias + offs_n, mask=offs_n < N, other=0.0)
        acc += bias_vals[None, :]

    tl.store(
        out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def low_resource_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    num_warps: int | None = None,
    num_stages: int = 3,
) -> torch.Tensor:
    """Small-batch BF16/FP16 linear with much lower per-CTA resources.

    This is intentionally conservative: it favors SM co-residency over peak GEMM
    speed. It is meant for decode-lane overlap experiments where the default
    cuBLASLt/nvjet kernel waits for whole-SM availability.
    """
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("low_resource_linear expects 2D x and 2D weight")
    if x.shape[1] != weight.shape[1]:
        raise ValueError(
            f"incompatible linear shapes: x={tuple(x.shape)}, weight={tuple(weight.shape)}"
        )
    if bias is not None and bias.shape[0] != weight.shape[0]:
        raise ValueError(
            f"incompatible bias shape: bias={tuple(bias.shape)}, weight={tuple(weight.shape)}"
        )
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("low_resource_linear is CUDA-only")

    M, K = x.shape
    N = weight.shape[0]
    if block_m is None or block_n is None or block_k is None or num_warps is None:
        auto_block_m, auto_block_n, auto_block_k, auto_num_warps = (
            _select_low_resource_linear_config(M, N)
        )
        block_m = auto_block_m if block_m is None else block_m
        block_n = auto_block_n if block_n is None else block_n
        block_k = auto_block_k if block_k is None else block_k
        num_warps = auto_num_warps if num_warps is None else num_warps

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _low_resource_linear_kernel[grid](
        x,
        weight,
        bias,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def _select_low_resource_linear_config(M: int, N: int) -> tuple[int, int, int, int]:
    # Decode lanes are small-M GEMMs. Keep the CTA footprint modest enough to
    # co-reside with capped slow-lane kernels, while avoiding unnecessary CTAs
    # for very small output projections.
    if M >= 16 and N >= 4096:
        return 16, 32, 128, 2
    if N <= 1024:
        return 8, 16, 128, 4
    return 8, 32, 128, 4
