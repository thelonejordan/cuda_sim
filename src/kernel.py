from typing import Callable
from dataclasses import dataclass
from itertools import product
from threading import Barrier
from concurrent.futures import ThreadPoolExecutor

import torch

from .helpers import timeit


torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
DTYPE = torch.float32


@dataclass
class dim3:
    z: int = 1
    y: int = 1
    x: int = 1
    @property
    def size(self): return self.z * self.y * self.x


def launch_cuda_kernel(gridSize: dim3, blockSize: dim3, kernel: Callable, shared_size: int=0):
    @timeit
    def dispatch_kernel(*kernargs):
        for blockIdx in product(range(gridSize.z), range(gridSize.y), range(gridSize.x)):
            shared_mem = torch.zeros(shared_size, dtype=DTYPE)
            barrier = Barrier(blockSize.size)
            with ThreadPoolExecutor(max_workers=blockSize.size) as e:
                for threadIdx in product(range(blockSize.z), range(blockSize.y), range(blockSize.x)):
                    e.submit(kernel, dim3(*blockIdx), dim3(*threadIdx), blockSize, barrier, shared_mem, *kernargs)
    return dispatch_kernel
