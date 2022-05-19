import sys

import torch
import numpy as np
import opt_einsum as oe
import sys

import time
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn as nn
import time

import math
import scipy
import torch
import torch.nn as nn

from torch import autograd
# Horace's FLOP counter
import torch

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten
from typing import List, Any, Dict, Union, Tuple
from numbers import Number
from collections import defaultdict

import pytest

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


flop_counts = defaultdict(lambda: defaultdict(int))
parents = ['Global']

def run_floptensor():
    print(torch.__version__)
    aten = torch.ops.aten

    def get_shape(i):
        return i.shape

    def prod(x):
        res = 1
        for i in x:
            res *= i
        return res

    def matmul_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
        """
        Count flops for matmul.
        """
        # Inputs should be a list of length 2.
        # Inputs contains the shapes of two matrices.
        input_shapes = [get_shape(v) for v in inputs]
        assert len(input_shapes) == 2, input_shapes
        assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
        flop = prod(input_shapes[0]) * input_shapes[-1][-1]
        return flop

    def addmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
        """
        Count flops for fully connected layers.
        """
        # Count flop for nn.Linear
        # inputs is a list of length 3.
        input_shapes = [get_shape(v) for v in inputs[1:3]]
        # input_shapes[0]: [batch size, input feature dimension]
        # input_shapes[1]: [batch size, output feature dimension]
        assert len(input_shapes[0]) == 2, input_shapes[0]
        assert len(input_shapes[1]) == 2, input_shapes[1]
        batch_size, input_dim = input_shapes[0]
        output_dim = input_shapes[1][1]
        flops = batch_size * input_dim * output_dim
        return flops

    def bmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
        """
        Count flops for the bmm operation.
        """
        # Inputs should be a list of length 2.
        # Inputs contains the shapes of two tensor.
        assert len(inputs) == 2, len(inputs)
        input_shapes = [get_shape(v) for v in inputs]
        n, c, t = input_shapes[0]
        d = input_shapes[-1][-1]
        flop = n * c * t * d
        return flop

    def conv_flop_count(
            x_shape: List[int],
            w_shape: List[int],
            out_shape: List[int],
            transposed: bool = False,
    ) -> Number:
        """
        Count flops for convolution. Note only multiplication is
        counted. Computation for addition and bias is ignored.
        Flops for a transposed convolution are calculated as
        flops = (x_shape[2:] * prod(w_shape) * batch_size).
        Args:
            x_shape (list(int)): The input shape before convolution.
            w_shape (list(int)): The filter shape.
            out_shape (list(int)): The output shape after convolution.
            transposed (bool): is the convolution transposed
        Returns:
            int: the number of flops
        """
        batch_size = x_shape[0]
        conv_shape = (x_shape if transposed else out_shape)[2:]
        flop = batch_size * prod(w_shape) * prod(conv_shape)
        return flop

    def conv_flop_jit(inputs: List[Any], outputs: List[Any]):
        """
        Count flops for convolution.
        """
        x, w = inputs[:2]
        x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))
        transposed = inputs[6]

        return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)

    def transpose_shape(shape):
        return [shape[1], shape[0]] + list(shape[2:])

    def conv_backward_flop_jit(inputs: List[Any], outputs: List[Any]):
        grad_out_shape, x_shape, w_shape = [get_shape(i) for i in inputs[:3]]
        output_mask = inputs[-1]
        fwd_transposed = inputs[7]
        flop_count = 0

        if output_mask[0]:
            grad_input_shape = get_shape(outputs[0])
            flop_count += conv_flop_count(grad_out_shape, w_shape, grad_input_shape, not fwd_transposed)
        if output_mask[1]:
            grad_weight_shape = get_shape(outputs[1])
            flop_count += conv_flop_count(transpose_shape(x_shape), grad_out_shape, grad_weight_shape, fwd_transposed)

        return flop_count

    flop_mapping = {
        aten.mm: matmul_flop_jit,
        aten.matmul: matmul_flop_jit,
        aten.addmm: addmm_flop_jit,
        aten.bmm: bmm_flop_jit,
        aten.convolution: conv_flop_jit,
        aten._convolution: conv_flop_jit,
        aten.convolution_backward: conv_backward_flop_jit,
    }

    flop_counts = defaultdict(lambda: defaultdict(int))
    parents = ['Global']

    def normalize_tuple(x):
        if not isinstance(x, tuple):
            return (x,)
        return x

    class FlopTensor(torch.Tensor):
        elem: torch.Tensor

        __slots__ = ['elem']

        @staticmethod
        def __new__(cls, elem):
            # The wrapping tensor (FlopTensor) shouldn't hold any
            # memory for the class in question, but it should still
            # advertise the same device as before
            r = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
                cls, elem.size(),
                strides=elem.stride(), storage_offset=elem.storage_offset(),
                # TODO: clone storage aliasing
                dtype=elem.dtype, layout=elem.layout,
                device=elem.device, requires_grad=elem.requires_grad
            )
            # ...the real tensor is held as an element on the tensor.
            r.elem = elem
            return r

        def __repr__(self):
            if self.grad_fn:
                return f"FlopTensor({self.elem}, grad_fn={self.grad_fn})"
            return f"FlopTensor({self.elem})"

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            def unwrap(e):
                return e.elem if isinstance(e, FlopTensor) else e

            print(func)

            # no_dispatch is only needed if you use enable_python_mode.
            # It prevents infinite recursion.
            rs = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            outs = normalize_tuple(rs)

            if func in flop_mapping:
                global flop_counts
                flop_count = flop_mapping[func](args, outs)
                for par in parents:
                    flop_counts[par][func.__name__] += flop_count

            def wrap(e):
                return FlopTensor(e) if isinstance(e, torch.Tensor) else e

            rs = tree_map(wrap, rs)
            return rs

    def create_backwards_push(name):
        class PushState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                global parents
                parents.append(name)
                return grad_outs

        return PushState.apply

    def create_backwards_pop(name):
        class PopState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                global parents
                assert (parents[-1] == name)
                parents.pop()
                return grad_outs

        return PopState.apply

    def enter_module(name):
        def f(module, inputs):
            global parents
            parents.append(name)
            inputs = normalize_tuple(inputs)
            out = create_backwards_pop(name)(*inputs)
            return out

        return f

    def exit_module(name):
        def f(module, inputs, outputs):
            global parents
            assert (parents[-1] == name)
            parents.pop()
            outputs = normalize_tuple(outputs)
            return create_backwards_push(name)(*outputs)

        return f

    def instrument_module(mod):
        for name, module in dict(mod.named_children()).items():
            print(f"Registering for module {name}, {module}")
            module.register_forward_pre_hook(enter_module(name))
            module.register_forward_hook(exit_module(name))

    def start_counting():
        global parents, flop_counts
        parents = ['Global']
        flop_counts.clear()

    def display_flops():
        for mod in flop_counts.keys():
            print(f"Module: ", mod)
            for k, v in flop_counts[mod].items():
                print(k, v)
            print()

    print(torch.__version__)

    from torch import autograd

    def create_linear(h, w):
        layer = nn.Linear(w, h, bias=False)
        layer.weight.data.copy_(2 * torch.ones(h, w))
        layer.weight.requires_grad_(False)
        return layer

    d = 10
    layer = [None] * 4
    layer[0] = create_linear(d, d)
    layer[1] = create_linear(d, d)
    layer[2] = create_linear(d, d)
    layer[3] = create_linear(1, d)
    net = torch.nn.Sequential(*layer)
    x = torch.ones(1, d)

    print("Forward pass")
    start_counting()
    instrument_module(net)
    x.requires_grad_(True)
    x = FlopTensor(x)
    y = net(x)
    display_flops()

    print("Backward pass")
    start_counting()
    autograd.grad(y, x, retain_graph=False)
    display_flops()

"""
def diag(A: FactoredTensor):
    pass

def trace(A: FactoredTensor):
    pass

def matmul(A: FactoredTensor):
    pass

def einsum(....):
    pass
"""

class FactoredTensor:
    name: str   # name of this Tensor
    tensors: Dict[str, Union["FactoredTensor", torch.Tensor]]   # {'A':torch.tensor[[1,2]]}
    indices: Tuple[str, ...]                                          # ('i','j','k')
    edges: Tuple[Tuple[str, Tuple[str, int]], ...]                       # ((('i',('A',0)), (('i',('A',1)), ('j', ('B',1)))
    atomic: bool    # whether this tensor is just a wrapper for PyTorch tensor. Wrapping is needed to enable contraction semantics

    # tensor initialized from atomic tensors. Composite tensors are constructed by wrappers like einsum
    def __init__(self, tensor:torch.Tensor, name=None):
        assert name is not None   # add name autogeneration later
        self.name = name
        self.tensors = {name: tensor}
        self.atomic = True

        if len(tensor.shape) == 1:
            self.indices = ('a',)
            self.edges = (('a',(name, 0)),)

        elif len(tensor.shape) == 2:
            self.indices = ('a', 'b')
            self.edges = (('a', (name, 0)), ('b', (name, 1)))

    @property
    def value(self):
        """Compute the value of this tensor"""
        if self.atomic:
            assert len(self.tensors) == 1
            return self.tensors[self.name]

    @property
    def flops(self):
        """Estimate flops needed to compute this tensor"""
        if self.atomic:
            return 0

    @property
    def shape(self):
        if self.atomic:
            return self.tensors[self.name].shape

    def _is_slot_node(self, node):
        return isinstance(node, tuple) and len(node) == 2

    def _find_child(self, node):
        """Each slot of component tensor maps to a single index, return this index"""

        # for now only implement children of slot nodes
        assert self._is_slot_node(node)
        children = []
        for idx, slot in self.edges:
            if slot == node:
                children.append(idx)
        assert len(children) == 1
        return children[0]

    def einsum_specs(self) -> List[Any]:
        """Returns list of einsum specs needed to compute this tensor and all of its component tensors.

         Nested matmul (A@B)@C corresponds to the following einsum computation
         out_1 = einsum('ab,bc->ac', A, B)
         out_0 = einsum('ab,bc->ac', out_1, C)

         The value returned here would be the following list
         [(('A','ab'), ('B','bc'), ('out_1','ac')),
          (('out_1','ab'), ('C','bc'), ('out_0','ac'))]

         Corresponding to the following einsum computation
        """

        ein_specs = []   # list of einsum specs for computing this tensor and all constituent tensors
        current_spec = []    # einsum spec for computing this tensor
        for tensor_name, tensor in self.tensors.items():
            tensor_indices = []
            for (dim_number, _) in enumerate(tensor.shape):
                slot = (tensor_name, dim_number)    #  ("A", i) refers to i'th dimension of tensor "A"
                tensor_indices.append(self._find_child(slot))  # each slot connects to a single index
            current_spec.append((tensor_name, ''.join(tensor_indices)))  # [('A', 'abc'), ...]

        if not self.atomic:
            tensor_name = self.name
            tensor_indices = []
            for (dim_number, _) in enumerate(self.shape):
                slot = (tensor_name, dim_number)
                tensor_indices.append(self._find_child(slot))
            current_spec.append((tensor_name, ''.join(tensor_indices)))  # [('A', 'abc'), ...]

        current_spec = tuple(current_spec)

        # recursively add child specs
        child_specs = []
        if not self.atomic:
            for tensor_name, tensor in self.tensors.items():
                if isinstance(tensor, FactoredTensor):
                    assert tensor_name == tensor.name
                    child_specs.extend(tensor.einsum_specs())

        return child_specs + [current_spec]


def test_atomic():
    A0 = torch.tensor([[1, 2], [3, 4.]])
    A = FactoredTensor(A0, name="A")
    assert A.einsum_specs() == [(('A', 'ab'),)]
    check_equal(A.value, A)
    assert A.flops == 0

    x0 = torch.tensor([1, 2.])
    x = FactoredTensor(x0, name="x")
    assert x.einsum_specs() == [(('x', 'a'),)]
    check_equal(x.value, x0)
    assert x.flops == 0

@pytest.mark.skip(reason="not ready")
def test_basic():
    A0 = torch.tensor([[1, 2], [3, 4.]])
    B0 = torch.tensor([[5, 6], [6, 7.]])
    x0 = torch.tensor([1, 2.])
    A = FactoredTensor(A0)
    B = FactoredTensor(B0)  # matrices have linear-operation semantics by default
    x = FactoredTensor(x0)  # vectors have (column)-vector semantics by default

    check_equal(x, x0)
    check_equal(x.T, x0)   # Transpose affects Tensor semantics, but array is unchanged in case of 1D

    check_equal(A @ x, A0 @ x0)
    with pytest.raises(Exception):  # 1d vectors have "column" vector semantics
        print(x @ A)
    check_equal(x.T @ A, x0 @ A0)
    check_equal(A.diag, torch.diag(A0))
    check_equal(A.trace, torch.trace(A0))

    ## check that optimization works. Also "test_outer_product" tests here
    d = 10
    x00 = torch.ones((d,))
    x = FactoredTensor(x00)
    B = x @ x.T
    assert diag(B).flops == 10
    assert B.flops == 100




    # register torch dispatch, then
    A = FlopTensor(torch.ones((3, 3)))
    B = torch.ones((3, 3))
    C = torch.ones((3, 1))
    A @ B @ C


if __name__ == '__main__':
    test_atomic()
    #  run_floptensor()
    sys.exit()
    # noinspection PyTypeChecker,PyUnreachableCode
    run_all_tests(sys.modules[__name__])
