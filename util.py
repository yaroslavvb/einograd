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


def check_close(observed, truth, rtol=1e-5, atol=1e-8, label: str = '') -> None:
    """Convenience method for check_equal with tolerances defaulting to typical errors observed in neural network
    ops in float32 precision."""
    return check_equal(observed, truth, rtol=rtol, atol=atol, label=label)


def check_equal(observed, truth, rtol=1e-9, atol=1e-12, label: str = '') -> None:
    """
    Assert fail any entries in two arrays are not close to each to desired tolerance. See np.allclose for meaning of rtol, atol

    """

    if hasattr(observed, 'value'):
        observed = observed.value

    if hasattr(truth, 'value'):
        truth = truth.value

    # special handling for lists, which could contain
    # if type(observed) == List and type(truth) == List:
    #    for a, b in zip(observed, truth):
    #        check_equal(a, b)

    truth = to_numpy(truth)
    observed = to_numpy(observed)

    # broadcast to match shapes if necessary
    if observed.shape != truth.shape:
        #        common_shape = (np.zeros_like(observed) + np.zeros_like(truth)).shape
        truth = truth + np.zeros_like(observed)
        observed = observed + np.zeros_like(truth)

    assert truth.shape == observed.shape, f"Observed shape {observed.shape}, expected shape {truth.shape}"
    # run np.testing.assert_allclose for extra info on discrepancies
    if not np.allclose(observed, truth, rtol=rtol, atol=atol, equal_nan=True):
        print(f'Numerical testing failed for {label}')
        np.testing.assert_allclose(truth, observed, rtol=rtol, atol=atol, equal_nan=True)


def create_linear(mat):
    mat = to_pytorch(mat)
    d1, d2 = mat.shape
    layer = nn.Linear(d1, d2, bias=False)
    layer.weight.data = mat
    return layer


def to_pytorch(x) -> torch.Tensor:
    """Convert numeric object to floating point PyTorch tensor."""
    return from_numpy(to_numpy(x))


def to_numpy(x, dtype: np.dtype = None) -> np.ndarray:
    """
    Convert numeric object to floating point numpy array. If dtype is not specified, use PyTorch default dtype.

    Args:
        x: numeric object
        dtype: numpy dtype, must be floating point

    Returns:
        floating point numpy array
    """

    assert np.issubdtype(dtype, np.floating), "dtype must be real-valued floating point"

    # Convert to normal_form expression from a special form (https://reference.wolfram.com/language/ref/Normal.html)
    if hasattr(x, 'normal_form'):
        x = x.normal_form()

    if type(x) == np.ndarray:
        assert np.issubdtype(x.dtype, np.floating), f"numpy type promotion not implemented for {x.dtype}"

    if hasattr(x, "detach"):
        dtype = pytorch_dtype_to_floating_numpy_dtype(x.dtype)
        return x.detach().cpu().numpy().astype(dtype)

    # list or tuple, iterate inside to convert PyTorch arrrays
    if type(x) in [list, tuple]:
        x = [to_numpy(r) for r in x]

    # Some Python type, use numpy conversion
    result = np.array(x, dtype=dtype)
    assert np.issubdtype(result.dtype, np.number), f"Provided object ({result}) is not numeric, has type {result.dtype}"
    if dtype is None:
        return result.astype(pytorch_dtype_to_floating_numpy_dtype(torch.get_default_dtype()))
    return result


def pytorch_dtype_to_floating_numpy_dtype(dtype):
    """Converts PyTorch dtype to numpy floating point dtype, defaulting to np.float32 for non-floating point types."""
    if dtype == torch.float64:
        dtype = np.float64
    elif dtype == torch.float32:
        dtype = np.float32
    elif dtype == torch.float16:
        dtype = np.float16
    else:
        dtype = np.float32
    return dtype


def from_numpy(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    else:
        return torch.tensor(x)

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

def freeze_multimap(idx_to_dim) -> Dict[Any, Tuple]:
    """Freezes dictionary {a->[], b->[]}"""
    # TODO(y) doesn't fully freeze since dictionaries are mutable
    d = {}
    for (key, value) in idx_to_dim.items():
        assert isinstance(value, List) or isinstance(value, Tuple)
        d[key] = tuple(value)
    return d


def run_all_tests(module):
    all_functions = inspect.getmembers(module, inspect.isfunction)
    for name, func in all_functions:
        if name.startswith("test_"):
            with timeit(name):
                func()
    print(module.__name__ + " tests passed.")
