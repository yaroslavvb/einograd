import inspect
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch import nn as nn


class LeastSquaresLoss(nn.Module):
    def __init__(self):
        super(LeastSquaresLoss, self).__init__()
        return

    # noinspection PyMethodMayBeStatic
    def forward(self, data, targets=None):
        if targets is None:
            targets = torch.zeros_like(data)

        if len(data.shape) == 1:
            err = data - targets
        else:
            err = data - targets.view(-1, data.shape[1])
        return torch.sum(err * err) / 2



global_forward_flops = 0
def get_global_forward_flops():
    global global_forward_flops
    return global_forward_flops

def increment_global_forward_flops(i):
    global global_forward_flops
    global_forward_flops += i

def reset_global_forward_flops():
    global global_forward_flops
    global_forward_flops = 0

class timeit:
    """Decorator to measure length of time spent in the block in millis and log
    it to TensorBoard."""

    def __init__(self, tag=""):
        self.tag = tag

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        interval_ms = 1000 * (self.end - self.start)
        print(f"{interval_ms:8.2f}   {self.tag}")


def run_all_tests(module):
    all_functions = inspect.getmembers(module, inspect.isfunction)
    for name, func in all_functions:
        if name.startswith("test_"):
            with timeit(name):
                func()
    print(module.__name__ + " tests passed.")
