from typing import Callable
from time import perf_counter


def cdiv(a: int, b: int): return (a + b - 1) // b  # equivalent to math.ceil()


def tidx(fidx: int, tw: int): return fidx // tw, fidx % tw            # tiledIdx from flatIdx
def fidx(tidx0: int, tidx1: int, tw: int): return tidx0 * tw + tidx1  # flatIdx from tiledIdx


class Counter:
    def __init__(self): self.loads, self.stores = 0, 0
    def increment(self, loads: int=0, stores: int=0):
        self.loads += loads    # increment loads
        self.stores += stores  # increment stores
    def show(self): print(f"Total reads: {self.loads} Total writes: {self.stores}")

def timeit(func: Callable):
    def timer(*args, **kwargs):
        st = perf_counter()
        out = func(*args, **kwargs)
        et = perf_counter()
        print(f"Time elapsed: {(et - st) * 1000:.2f} ms")
        return out
    return timer
