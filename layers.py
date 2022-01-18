import util
from base import *

from util import *

import torch.nn.functional as F

_idx0 = 'a'


class ContractibleTensor(Tensor, ABC):
    """
    Tensor is split into 2, abstract API is in Tensor, reusable implementation details are in
    ContractibleTensor
    """

    @property
    def out_dims(self) -> Tuple[int]:
        return 42,

    @property
    def in_dims(self) -> Tuple[int]:
        return 42,

    def out_idx(self):
        upper = ''
        offset = 0
        # TODO(y): Covector does not have out_dims defined, test what happens
        # TODO(y): replace out_dims with property call
        if hasattr(self, 'out_dims'):
            upper = ''.join(chr(ord(_idx0) + offset + i) for i in range(len(self.out_dims)))
        return upper

    def in_idx(self):
        # TODO(y) replace with properties
        lower = ''
        offset = 0
        if hasattr(self, 'in_dims'):
            lower = ''.join(chr(ord(_idx0) + offset + i) for i in range(len(self.in_dims)))
        return lower

    def all_idx(self, offset=0):
        """Generate string corresponding to upper,lower indices, with offset.
        IE for tensor with 2 upper indices 'ab' and 1 lower index 'c'
        IE, offset 0: "abc"
            offset 1: "bcd", etc"""

        upper, lower = '', ''
        if hasattr(self, 'out_dims'):
            upper = ''.join(chr(ord(_idx0) + offset + i) for i in range(len(self.out_dims)))

        offset += len(upper)
        if hasattr(self, 'in_dims'):
            lower = ''.join(chr(ord(_idx0) + offset + i) for i in range(len(self.in_dims)))
        return upper + lower, upper, lower

    def __mul__(self, other):
        print('other is ', other)
        assert isinstance(other, ContractibleTensor), f"contracting tensor with {type(other)}"

        t1 = self
        t2 = other

        # assert isinstance(t1, DenseLinear) and isinstance(t2, DenseVector), "contraction tested only for
        # matrix@vector " return self.W * x

        assert t1.in_dims == t2.out_dims  # ij,j -> i
        (t1_idx, t1_idx_out, t1_idx_in) = t1.all_idx()
        (t2_idx, t2_idx_out, t2_idx_in) = t2.all_idx(offset=len(t1.out_idx()))

        t1_set = set(t1_idx)
        t2_set = set(t2_idx)
        contracted_set = t1_set.intersection(t2_set)
        result_set = t1_set.union(t2_set).difference(contracted_set)
        result_idx = ''.join(sorted(list(result_set)))

        einsum_str = f"{t1_idx},{t2_idx}->{result_idx}"
        print('doing einsum ', einsum_str)
        print('tensor 1', t1.value)
        print('tensor 2', t2.value)
        data = torch.einsum(einsum_str, t1.value, t2.value)

        # figure out new input, output indices, create corresponding object
        out_idx = result_set.intersection(set(t1_idx_out + t2_idx_out))
        in_idx = result_set.intersection(set(t1_idx_in + t2_idx_in))
        if out_idx and not in_idx:
            assert len(out_idx) == 1, "don't support multi-index yet"
            return DenseVector(data)
        elif in_idx and not out_idx:
            assert len(in_idx) == 1, "don't support multi-index yet"
            return DenseCovector(data)
        elif in_idx and out_idx:
            assert len(in_idx) == len(out_idx) == 1, "don't support multi-index yet"
            return DenseLinear(data)
        elif not in_idx and not out_idx:
            assert data.shape == ()
            return DenseScalar(data)

    def __str__(self):
        return str(self.value)


class ZeroTensor(Tensor):
    """Arbitrary shape tensor of zeros"""

    def in_dims(self):
        return ()

    def out_dims(self):
        return ()


zero = ZeroTensor()


class DenseScalar(Scalar):
    def __init__(self, value):
        value = to_pytorch(value)
        assert len(value.shape) == 0
        self._value = value

    @property
    def in_dims(self):
        return ()

    @property
    def out_dims(self):
        return ()

    @property
    def value(self):
        return self._value


class DenseVector(Vector, ContractibleTensor):
    _value: torch.Tensor
    _out_dims: Tuple[int]

    def __init__(self, value):
        value = to_pytorch(value)
        assert len(value.shape) == 1
        assert value.shape[0] > 0
        self._out_dims = value.shape
        self._value = value

    @property
    def in_dims(self):
        return ()

    @property
    def out_dims(self):
        return self._out_dims

    @property
    def value(self):
        return self._value

    @property
    def T(self):
        return DenseCovector(self._value)


class DenseCovector(Covector, ContractibleTensor):
    _value: torch.Tensor
    _in_dims: Tuple[int]

    def __init__(self, value):
        value = to_pytorch(value)
        assert len(value.shape) == 1
        assert value.shape[0] > 0
        self._in_dims = value.shape
        self._value = value

    @property
    def in_dims(self):
        return self._in_dims

    @property
    def out_dims(self):
        return ()

    @property
    def value(self):
        return self._value

    @property
    def T(self):
        return DenseCovector(self._value)


class DenseLinear(LinearMap, ContractibleTensor):
    def __init__(self, value):
        value = to_pytorch(value)
        assert len(value.shape) == 2
        assert value.shape[0] > 0
        assert value.shape[1] > 0
        self._out_dims = (value.shape[0],)
        self._in_dims = (value.shape[1],)
        self._value = value

    @property
    def out_dims(self) -> Tuple[int]:
        return self._out_dims

    @property
    def in_dims(self) -> Tuple[int]:
        return self._in_dims

    @property
    def value(self):
        return self._value


class LeastSquares(AtomicFunction):
    """Least squares loss"""

    def __init__(self, dim: int):
        self._in_dims = (dim,)
        self._out_dims = ()

    def __call__(self, x: Tensor):
        x = x.value
        return DenseScalar((x * x).sum() / 2)

    @property
    def d(self):
        return DLeastSquares(dim=self._in_dims[0])

    @property
    def in_dims(self):
        return self._in_dims

    @property
    def out_dims(self):
        return self._out_dims

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            return MemoizedFunctionComposition([self, other])
        else:
            return NotImplemented


class DLeastSquares(AtomicFunction, LinearizedFunction):
    """Derivatives of cross entropy"""

    def __init__(self, dim: int, order: int = 1):
        self._in_dims = (dim,)
        self._out_dims = ()
        self.order = order

    def __call__(self, x: DenseVector):
        assert self.order <= 2, "third and higher order derivatives not implemented"
        n = self._in_dims[0]

        if self.order == 1:
            return x.T
        elif self.order == 2:
            return DenseLinear(torch.eye(n))
        # three-dimensional identity tensor, does not exist in numpy
        elif self.order == 3:
            assert False, "TODO: wrap this into proper rank-3 tensor"
            # x = torch.einsum('ij,jk->ijk', torch.eye(n), torch.eye(n))

    @property
    def d(self):
        return DLeastSquares(self._in_dims[0], self.order + 1)

    @property
    def in_dims(self):
        return self._in_dims

    @property
    def out_dims(self):
        return self._out_dims

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            return MemoizedFunctionComposition([self, other])
        else:
            return NotImplemented


class Identity(AtomicFunction):
    def __init__(self, dim: int):
        self._in_dims = (dim,)
        self._out_dims = (dim,)

    def __call__(self, x: Tensor):
        return x

    @property
    def d(self):
        return DIdentity(self._in_dims[0])

    @property
    def in_dims(self):
        return self._in_dims

    @property
    def out_dims(self):
        return self._out_dims

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            return MemoizedFunctionComposition([self, other])
        else:
            return NotImplemented


class DIdentity(AtomicFunction):
    """Derivatives of identity"""

    def __init__(self, dim: int, order: int = 1):
        self._in_dims = (dim,)
        self._out_dims = (dim,)
        self.order = order

    @property
    def out_dims(self):
        return self._out_dims

    @property
    def in_dims(self):
        return self._in_dims

    def d(self):
        return zero

    def __call__(self, x: DenseVector):
        assert self.order <= 2, "third and higher order derivatives not implemented"
        n = self._in_dims[0]

        if self.order == 1:
            return DenseLinear(torch.eye(n))
        elif self.order == 2:
            return 0

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            return MemoizedFunctionComposition([self, other])
        else:
            return NotImplemented


class Relu(AtomicFunction):
    """One dimensional relu"""

    def __init__(self, dim: int):
        self._in_dims = (dim,)
        self._out_dims = (dim,)

    def __call__(self, x: Tensor):
        x = x.value
        return DenseVector(F.relu(x))

    @property
    def d(self):
        return DRelu(self._in_dims[0])

    @property
    def in_dims(self):
        return self._in_dims

    @property
    def out_dims(self):
        return self._out_dims

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            return MemoizedFunctionComposition([self, other])
        else:
            return NotImplemented


class DRelu(AtomicFunction, LinearizedFunction):
    """Derivatives of relu"""

    def __init__(self, dim: int, order: int = 1):
        self._in_dims = (dim,)
        self._out_dims = (dim,)
        self.order = order

    @property
    def out_dims(self):
        return self._out_dims

    @property
    def in_dims(self):
        return self._in_dims

    def d(self):
        return zero

    def __call__(self, x: Tensor) -> DenseLinear:
        x = x.value
        return DenseLinear(torch.diag((x > torch.tensor(0)).float()))

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            return MemoizedFunctionComposition([self, other])
        else:
            return NotImplemented


# class ContractibleTensor:
#     """Including this operation implements * contraction"""
#
#     def __mul__(self, other):
#         """Contraction operation"""
#         assert isinstance(other, ContractibleTensor)
#
#         # assert isinstance(x, DenseVector)
#         # assert self.in_dims == x.out_dims
#         # in1_idx = self.all_idx()
#         # in2_idx = x.all_idx(offset=len(self.in_idx()))
#         # out_idx = self.out_idx()
#         # einsum_str = f"{in1_idx},{in2_idx}->{out_idx}"
#         # print('doing einsum ', einsum_str)
#         # data = torch.einsum(einsum_str, self.data, x.data)
#         # return DenseVector(data)


#    TODO(y): maybe also implement Function interface?
class MemoizedFunctionComposition:
    """Represents a composition of functions with memoization
    Unbound call, ie f@g@h, can be used as intermediate result for constructing a composition
    Bound call, ie (f@g@h)(x), at this point it is frozen and can't be modified.
    """

    children: List[Any]  # List[Function] fix: forward references
    # parent               # FunctionComposition type, multiple Compositions point here. fix: forward reference
    arg: Any

    def __init__(self, children, parent=None):
        self.arg = None
        self.parent = parent
        self.children = children
        self._saved_outputs = [None] * (len(children) + 1)

        for child in children:
            if hasattr(child, 'parent') and child.parent is not None:
                assert False, f"Warning, Node {child} already has parent {child.parent}"
                # print(f"Warning, Node {child} already has parent {child.parent}, reuse the same function object in "
                #      f"multiple places in tree may break memoization.")
            child.parent = self

    def __matmul__(self, other):
        assert self.arg is None, "Can't combine compositions with bound parameters"
        if isinstance(other, Function):
            return MemoizedFunctionComposition(self.children + [other])
        else:
            return NotImplemented

    # only allow simple slicing
    def __getitem__(self, s):
        if isinstance(s, slice):
            if isinstance(s.step, int):
                assert s.step == 1
            error_msg = "this case hasn't been tested, for now only single level of parent  redirection is allowed"
            assert self.parent is None, error_msg
            backlink = self if self.parent is None else self.parent
            assert s.stop is None
            assert len(self.children[s.start:]) > 0
            return MemoizedFunctionComposition(self.children[s.start:], backlink)
        else:
            assert isinstance(s, int)
            return self.children[s]

    def _bind(self, arg):
        print('binding ', arg)
        self.arg = arg
        self._saved_outputs[len(self.children)] = arg

    def memoized_compute(self, node):
        """Composition([h3,h2,h1]): memoized_compute(h2) computes everything up to h2"""
        assert self.arg is not None, "arg not bound, call _bind first"
        assert id(self._saved_outputs[len(self.children)]) == id(self.arg)
        assert node in self.children, "Trying to compute {node} but it's not in Composition"
        idx = self.children.index(node)
        print('memoized compute on ', idx)

        # last_saved gives position of function whose output has been cached
        # we treat "x" as a dummy function which returns itself, so
        # last_cached == len(children) means just the output of "x" was cached
        for last_cached in range(len(self.children)):
            if self._saved_outputs[last_cached] is not None:
                break
        else:
            last_cached = len(self.children)

        print(f'found saved output of {last_cached} node')
        for i in range(last_cached - 1, -1, -1):
            if i == len(self.children):
                assert id(self._saved_outputs[last_cached]) == id(self.arg)
                continue

            util.increment_global_forward_flops(1)
            result = self.children[i](self._saved_outputs[i + 1])
            self._saved_outputs[i] = result
            print('saving output of ', i)

        return self._saved_outputs[idx]

    def __call__(self, arg: Vector):
        assert isinstance(arg, Vector), "must call function with Vector type"
        if self.parent is not None:
            assert isinstance(self.parent, MemoizedFunctionComposition)
            assert id(arg) == id(self.parent.arg)
            return self.parent.memoized_compute(self.children[0])

        if self.arg is None:
            self._bind(arg)
        else:
            assert id(self.arg) == id(arg), "Reusing same composition for multiple args"

        return self.memoized_compute(self.children[0])


class LinearLayer(AtomicFunction):
    """Dense Linear Layer"""

    _out_dims: Tuple[int]
    _in_dims: Tuple[int]
    W: DenseLinear

    def __init__(self, W):
        W = to_pytorch(W)
        assert len(W.shape) == 2
        assert W.shape[0] >= 1
        assert W.shape[1] >= 1
        W = DenseLinear(W)
        self._out_dims = W.out_dims
        self._in_dims = W.in_dims
        self.W = W

    @property
    def d(self):
        return DLinearLayer(self.W)

    @property
    def out_dims(self) -> Tuple[int]:
        return self._out_dims

    @property
    def in_dims(self) -> Tuple[int]:
        return self._in_dims

    def __call__(self, x: Vector) -> DenseVector:
        assert isinstance(x, Vector)
        result = self.W * x
        assert isinstance(result, DenseVector)
        return result


class DLinearLayer(AtomicFunction, LinearizedFunction):
    """derivative of Dense Linear Layer"""

    W: DenseLinear

    def in_dims(self):
        return self.W.in_dims

    def out_dims(self):
        return self.W.out_dims

    def __init__(self, W: DenseLinear):
        # for now, only support matrices
        assert len(W.in_idx()) == 1
        assert len(W.out_idx()) == 1
        self.W = W

    def __call__(self, _unused_x: Tensor) -> DenseLinear:
        return self.W

    def d(self):
        return self

# creator helper methods
#    W = make_linear([[1, -2], [-3, 4]])
#    U = make_linear([[5, -6], [-7, 8]])
#    x0 = make_vector([1, 2])
#    nonlin = make_sigmoid(x0)
#    loss = make_xent(x0)   # x0 used for shape inference

# def make_linear(mat):
#    mat = to_pytorch(mat)
#    d1, d2 = mat.shape
#    layer = nn.Linear(d1, d2, bias=False)
#    layer.weight.data = mat
#    return layer
