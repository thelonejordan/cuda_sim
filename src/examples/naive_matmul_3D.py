from typing import Tuple
from threading import Barrier

import torch
from torch import Tensor

from src.kernel import launch_cuda_kernel, dim3, DTYPE
from src.helpers import Counter, fidx, tidx, cdiv


def matmul_naive3D_kernel(blockIdx: dim3, threadIdx: dim3, blockSize: dim3, barrier: Barrier, shared_mem: Tensor,
                          buffers: Tuple[Tensor, ...], metadata: Tuple[int, ...], counter: Counter):
    m = (blockIdx.z * blockSize.z) + threadIdx.z
    n = (blockIdx.y * blockSize.y) + threadIdx.y
    k = (blockIdx.x * blockSize.x) + threadIdx.x
    (C, A, B), (M, K, N) = buffers, metadata
    if m < M and n < N and k < K:
        C[fidx(m, n, N)] += A[fidx(m, k, K)] * B[fidx(k, n, N)]
        counter.increment(loads=2, stores=1)


def matmul_naive3D(A: Tensor, B: Tensor, tileWidth: int):
    (M, K), (K_, N) = A.shape, B.shape
    assert K == K_, f"inner dims should match, {K} != {K_}"
    A = A.contiguous() if not A.is_contiguous() else A
    B = B.contiguous() if not B.is_contiguous() else B
    C = torch.empty(M, N, dtype=DTYPE)

    gridSize = dim3(cdiv(M, tileWidth), cdiv(N, tileWidth), cdiv(K, tileWidth))
    blockSize = dim3(tileWidth, tileWidth, tileWidth)
    kernel = launch_cuda_kernel(gridSize, blockSize, matmul_naive3D_kernel)
    kernel((C.flatten(), A.flatten(), B.flatten()), (M, K, N), counter:=Counter())
    counter.show()
    return C


if __name__ == "__main__":
    # test example
    M, K, N = 12, 24, 18
    A = torch.randn(M, K, dtype=DTYPE)
    B = torch.randn(K, N, dtype=DTYPE)
    C_ref = A @ B

    def check_output(C: Tensor):
        std = torch.std(C - C_ref).item()
        ret = "PASSED" if torch.allclose(C, C_ref, atol=1e-6) else f"FAILED"
        return f"{ret} {std=:.4f}"

    # print(check_output((A.view(M, 1, K) * B.T.view(1, N, K)).sum(dim=-1)))

    C = matmul_naive3D(A, B, 3)
    print(check_output(C))
