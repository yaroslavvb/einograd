from typing import Dict

import more_itertools
import natsort
import torch.nn.functional as F

from opt_einsum import helpers as oe_helpers
import opt_einsum as oe
# import natsort

import util
from base import *
from util import *

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


class IdentityLinearMap(Tensor):
    """tensor representing identity linear map"""

    def in_dims(self):
        return ()

    def out_dims(self):
        return ()


zero = ZeroTensor()
Zero = ZeroTensor()  # TODO(y) remove one of the zeros
One = IdentityLinearMap()


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


class DenseSymmetricBilinear(SymmetricBilinearMap, ContractibleTensor):
    """Symmetric bilinear map represented with a rank-3 tensor"""

    def __init__(self, value):
        value = to_pytorch(value)
        assert len(value.shape) == 3
        assert value.shape[0] > 0
        assert value.shape[1] > 0
        assert value.shape[2] > 0
        assert value.shape[1] == value.shape[2]
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


class DenseQuadraticForm(QuadraticForm, ContractibleTensor):
    """Symmetric bilinear map represented with a rank-2 tensor"""

    def __init__(self, value):
        value = to_pytorch(value)
        assert len(value.shape) == 2
        assert value.shape[0] > 0
        assert value.shape[1] > 0
        assert value.shape[0] == value.shape[1]
        self._out_dims = ()
        self._in_dims = (value.shape[1],)
        self._value = value

    @property
    def out_dims(self) -> tuple:
        return self._out_dims

    @property
    def in_dims(self) -> Tuple[int]:
        return self._in_dims

    @property
    def value(self):
        return self._value


class DenseLinear(LinearMap, ContractibleTensor):
    """Symmetric linear map represented with a rank-2 tensor"""

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

    def d(self, order=1):
        return DLeastSquares(dim=self._in_dims[0], order=order)

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
    """Derivatives of LeastSquares"""

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
            return DenseQuadraticForm(torch.eye(n))
        # three-dimensional identity tensor, does not exist in numpy
        elif self.order == 3:
            assert False, "TODO: wrap this into proper rank-3 tensor"
            # x = torch.einsum('ij,jk->ijk', torch.eye(n), torch.eye(n))

    @property
    def d1(self):
        return self.d(1)

    def d(self, order=1):
        return DLeastSquares(dim=self._in_dims[0], order=self.order + order)

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

    def d(self, order=1):
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

    @property
    def d1(self):
        return self.d(1)

    def d(self, order=1):
        if order == 1:
            return One
        elif order >= 2:
            return Zero

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

    def d(self, order=1):
        return DRelu(self._in_dims[0], order=order)

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

    def d(self, order=1):
        return Zero

    def __call__(self, x: Tensor) -> DenseLinear:
        if self.order == 1:
            x = x.value
            return DenseLinear(torch.diag((x > torch.tensor(0)).float()))
        assert False

    def __matmul__(self, other):
        if isinstance(other, AtomicFunction):
            return MemoizedFunctionComposition([self, other])
        else:
            return NotImplemented


class Sigmoid(AtomicFunction):
    """One dimensional relu"""

    def __init__(self, dim: int):
        self._in_dims = (dim,)
        self._out_dims = (dim,)

    def __call__(self, x: Tensor):
        x = x.value
        return DenseVector(torch.sigmoid(x))

    def d(self, order=1):
        return DSigmoid(self._in_dims[0], order=order)

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


class DSigmoid(AtomicFunction, LinearizedFunction):
    """Derivatives of sigmoid"""

    def __init__(self, dim: int, order: int = 1):
        assert order >= 1
        self._in_dims = (dim,)
        self._out_dims = (dim,)
        self.order = order

    @property
    def out_dims(self):
        return self._out_dims

    @property
    def in_dims(self):
        return self._in_dims

    def d(self, order=1):
        return DSigmoid(self._in_dims[0], order=self.order + order)

    def __call__(self, x: Tensor) -> ContractibleTensor:
        x = x.value
        s = torch.sigmoid(x)
        if self.order == 1:
            return DenseLinear(torch.diag(s * (1 - s)))
        elif self.order == 2:
            n = self._in_dims[0]
            p = s * (1 - s) * (1 - 2 * s)
            eye_3 = torch.einsum('ij, jk -> ijk', torch.eye(n), torch.eye(n))
            diag_3_p = torch.einsum('ijk, k -> ijk', eye_3, p)
            return DenseSymmetricBilinear(diag_3_p)

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

    def d(self, order=1):
        return DLinearLayer(self.W, order=order)

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

    def __init__(self, W: DenseLinear, order=1):
        # for now, only support matrices
        self.order = order
        assert len(W.in_idx()) == 1
        assert len(W.out_idx()) == 1
        self.W = W

    def __call__(self, _unused_x: Tensor) -> DenseLinear:
        return self.W

    def d(self, order=1):
        if order == 1:
            return self
        else:
            return Zero


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

        print('initializing with ', children, parent)
        # if creating a sub-composition, extra sanity check that the nodes we are using
        # are already pointing to the parent composition
        for child in children:
            if parent:
                assert child.parent == parent
            else:
                if hasattr(child, 'parent') and child.parent is not None:
                    assert False, f"Warning, Node {child} already has parent {child.parent}"
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
            assert len(self.children[s.start:]) > 0, f"only have {len(self.children)} members of composition, " \
                                                     f"attempted to start at {s.start} "
            return MemoizedFunctionComposition(self.children[s.start:], backlink)
        else:
            assert False, "use [:] slicing as [i] is ambiguous"
            # assert isinstance(s, int)
            # return self.children[s]

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
        print(f'memoized compute on {idx}, {node}')

        # last_saved gives position of function whose output has been cached
        # we treat "x" as a dummy function which returns itself, so
        # last_cached == len(children) means just the output of "x" was cached
        for last_cached in range(len(self.children)):
            if self._saved_outputs[last_cached] is not None:
                break
        else:
            last_cached = len(self.children)

        print(f'found saved output of {last_cached} node')
        for i in range(last_cached - 1, idx - 1, -1):
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
            if self.parent.arg is not None:
                assert id(arg) == id(self.parent.arg)
            else:
                self.parent._bind(arg)
            return self.parent.memoized_compute(self.children[0])

        if self.arg is None:
            self._bind(arg)
        else:
            assert id(self.arg) == id(arg), "Reusing same composition for multiple args"

        return self.memoized_compute(self.children[0])


class StructuredTensor(Tensor):
    # tensor in a structured form (einsum)
    # it supports lazy contraction with other tensors, calculating flop counts
    # performing the calculation

    _in_dims: Tuple[int]
    _out_dims: Tuple[int]

    out_indices: List[chr]
    in_indices: List[chr]
    contracted_indices: List[chr]

    _index_spec_list: List[str]  # ['ij|k', 'k|lm'] => [output1|input1,output2|input2]
    _einsum_spec: str            # 'ij,jk->ik'

    tensors: List[torch.Tensor]  # [torch.ones((2,2,2)), torch.ones((2,2,2))]

    index_degree: Dict[chr, int]
    """for each index, count how many tensors share this index
    it is the degree of the "hyper-edge" labeled by this index in the Tensor Network Diagram
    d['i']==1 indicates a free index, i is a dangling edge
    d['i']==2 indicates contraction of two tensors, i is regular edge 
    d['i']==3 indicates contraction of three tensors, i is a hyper-edge connecting three tensors """

    # extra debugging, for each index, keep count of how many tensors have this index as out/in index
    # as well as the list of tensors
    index_out_degree: Dict[chr, int]
    index_in_degree: Dict[chr, int]
    index_out_tensors: Dict[chr, List[torch.Tensor]]  # d['i'] == [tensor1, tensor2, ...]
    index_in_tensors: Dict[chr, List[torch.Tensor]]

    index_dim: Dict[chr, int]
    "index dimensions, ie index_dim['i']==3"

    @property
    def in_dims(self):
        return self._in_dims

    @property
    def out_dims(self):
        return self._out_dims

    def __init__(self, index_spec_list, tensors):
        """['ij|k', 'k|lm'], [tensor1, tensor2]"""
        if len(index_spec_list) != len(tensors):
            print(f"Provided {len(tensors)} tensors, but your index spec has {len(index_spec_list)} terms: ")
            for (i, term) in enumerate(index_spec_list):
                print(f"term {i:2d}: {term:>20}")
                assert False

        self._index_spec_list = index_spec_list
        self.tensors = tensors
        self.index_degree = {}
        self.index_out_degree = {}
        self.index_in_degree = {}
        self.index_out_tensors = {}
        self.index_in_tensors = {}

        all_indices = set()   # all

        # create dict of sizes, by matching indices to tensors
        index_dim = {}  # ['ij'], [np.ones((2,5))] gives {'i': 2, 'j': 5}
        for (index_spec, tensor) in zip(index_spec_list, tensors):
            assert isinstance(index_spec, str)
            assert isinstance(tensor, torch.Tensor), f"Provided not an instance of torch.Tensor, {index_spec}, {tensor}"
            output_indices, input_indices = index_spec.split('|')
            all_indices_tensor = output_indices + input_indices

            assert len(all_indices_tensor) == len(set(all_indices_tensor))
            if gl.PURE_TENSOR_NETWORKS:  # this disallows diagonal tensors
                assert not set(input_indices).intersection(set(output_indices))

            all_indices.update(set(all_indices_tensor))

            for idx in output_indices:
                # noinspection PyTypeChecker
                self.index_out_tensors.setdefault(idx, []).append(tensor)
                self.index_out_degree[idx] = self.index_out_degree.get(idx, 0) + 1
            for idx in input_indices:
                # noinspection PyTypeChecker
                self.index_in_tensors.setdefault(idx, []).append(tensor)
                self.index_in_degree[idx] = self.index_in_degree.get(idx, 0) + 1

            for idx in set(all_indices_tensor):
                self.index_degree[idx] = self.index_degree.get(idx, 0) + 1

            for (idx, dim) in zip(all_indices_tensor, tensor.shape):
                if idx in index_dim:
                    assert index_dim[idx] == dim, f"trying to set idx {idx} in indices {index_spec} to {dim}, " \
                                                  f"but it's already set to have dimension {index_dim[idx]}"
                assert dim > 0, f"Index {idx} has dimension {dim}"
                index_dim[idx] = dim
        self.index_dim = index_dim

        # sanity check, for each index make sure it appears equal number of times as contravariant and covariant
        self.contracted_indices = []
        self.out_indices = []
        self.in_indices = []
        for idx in all_indices:
            # number of tensors for which this idx is upper/contravariant
            out_count = len(self.index_out_tensors.get(idx, []))
            # number of tensors for which this idx is lower/covariant
            in_count = len(self.index_in_tensors.get(idx, []))
            assert out_count == self.index_out_degree.get(idx, 0)
            assert in_count == self.index_in_degree.get(idx, 0)

            if out_count and in_count:
                assert out_count == in_count
                if gl.PURE_TENSOR_NETWORKS:
                    assert out_count == 1  # in pure tensor networks, each index is contracted at most once
                else:
                    assert out_count <= 2, f"Index {idx} is contravariant in {out_count} tensors, suspicious," \
                                           f"it should be 1 for regular tensors, and 2 for diagonal matrices "
                assert idx not in self.contracted_indices, f"Trying to add {idx} as contracted index twice"
                self.contracted_indices.append(idx)
            elif out_count and not in_count:
                assert idx not in self.out_indices, f"Trying to add {idx} as output index twice"
                self.out_indices.append(idx)
            elif in_count and not out_count:
                assert idx not in self.out_indices, f"Trying to add {idx} as input index twice"
                self.in_indices.append(idx)
            else:
                assert False, f"Shouldn't be here, {idx} is marked as occuring {out_count} times as contravariant " \
                              f"and {in_count} as covariant"

        assert len(self.out_indices) == len(set(self.out_indices))
        assert len(self.in_indices) == len(set(self.in_indices))
        assert not set(self.out_indices).intersection(self.in_indices)

        self._out_dims = tuple(self.index_dim[c] for c in self.out_indices)
        self._in_dims = tuple(self.index_dim[c] for c in self.in_indices)

        einsum_in = ','.join(index_spec.replace('|', '') for index_spec in self._index_spec_list)
        einsum_out = ''.join(self.out_indices) + ''.join(self.in_indices)
        self._einsum_spec = f'{einsum_in}->{einsum_out}'

    @staticmethod
    def from_dense_vector(x: torch.Tensor):
        """Creates StructuredTensor object corresponding to given dense vector"""
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 1
        assert x.shape[0] > 0
        return StructuredTensor(['i|'], [x])

    @staticmethod
    def from_dense_covector(x: torch.Tensor):
        """Creates StructuredTensor object corresponding to given dense covector"""
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 1
        assert x.shape[0] > 0
        return StructuredTensor(['|i'], [x])

    @staticmethod
    def from_dense_matrix(x: torch.Tensor):
        """Creates StructuredTensor object (LinearMap with 1 output, 1 input indices) from given matrix
        """
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 2
        assert x.shape[0] > 0
        assert x.shape[1] > 0
        return StructuredTensor(['i|j'], [x])

    # @staticmethod(x)

    import more_itertools

    def rename_index(self, old_name, new_name, tag='none'):
        print(f"naming {tag}:{old_name} to {new_name}")

        def rename_dictionary_entry(d: Dict[chr, Any], old_name: chr, new_name: chr):
            if old_name not in d:
                return
            assert isinstance(d, dict)
            assert new_name not in d
            assert isinstance(old_name, str)
            assert len(old_name) == 1
            assert isinstance(new_name, str)
            assert len(new_name) == 1
            d[new_name] = d[old_name]
            del d[old_name]

        def rename_list_entry(l, old_name, new_name):  # {len(l.count(old_name)}!=1
            if old_name not in l:
                return
            assert isinstance(l, list)
            assert isinstance(old_name, str)
            assert len(old_name) == 1
            assert isinstance(new_name, str)
            assert len(new_name) == 1

            # assert l.count(old_name) == 1, f"Found  {l.count(old_name)} instances of {old_name} in {l}"
            pos = l.index(old_name)
            l[pos] = new_name

        rename_list_entry(self.out_indices, old_name, new_name)
        rename_list_entry(self.in_indices, old_name, new_name)
        rename_list_entry(self.contracted_indices, old_name, new_name)
        # _index_spec_list: List[str]  # ['ij|k', 'k|lm'] => [output1|input1,output2|input2]
        for i, index_spec in enumerate(self._index_spec_list):
            self._index_spec_list[i] = index_spec.replace(old_name, new_name)
        #  _einsum_spec: str  # 'ij,jk->ik'
        self._einsum_spec = self._einsum_spec.replace(old_name, new_name)
        rename_dictionary_entry(self.index_degree, old_name, new_name)
        rename_dictionary_entry(self.index_out_degree, old_name, new_name)
        rename_dictionary_entry(self.index_in_degree, old_name, new_name)
        rename_dictionary_entry(self.index_out_tensors, old_name, new_name)
        rename_dictionary_entry(self.index_in_tensors, old_name, new_name)
        rename_dictionary_entry(self.index_dim, old_name, new_name)
        rename_list_entry(self.out_indices, old_name, new_name)
        rename_list_entry(self.in_indices, old_name, new_name)
        rename_list_entry(self.contracted_indices, old_name, new_name)

    def _check_indices_sorted(self):
        assert more_itertools.is_sorted(self.out_indices, strict=True)
        assert more_itertools.is_sorted(self.contracted_indices, strict=True)
        assert more_itertools.is_sorted(self.in_indices, strict=True)
        assert more_itertools.is_sorted(self.out_indices + self.contracted_indices + self.in_indices, strict=True)

        # check that output/input indices are consecutive
        if self.out_indices:
            assert self.out_indices[-1] == chr(ord(self.out_indices[0])+len(self.out_indices)-1)
        if self.in_indices:
            assert self.in_indices[-1] == chr(ord(self.in_indices[0])+len(self.in_indices)-1)

    def contract(self, other: 'StructuredTensor'):
        """This modifies both current and other tensor"""

        print('my old spec list', self._index_spec_list)
        print('other old spec list', other._index_spec_list)

        # relabeling invariants
        # self.input_indices are larger than any other indices
        # other.contracted_indices + output_indices are larger than input indices
        #
        # increment all indices of "other" to make all k "other" input indices match first k self "output" indices
        # increment remaining self.output indices to be larger than largest input index.

        # is_sorted:

        self._check_indices_sorted()
        other._check_indices_sorted()

        left = StructuredTensor(self._index_spec_list, self.tensors)
        right = StructuredTensor(other._index_spec_list, other.tensors)

        # first increment (never decrement because of _check_indices_sorted invariant) indices on the right to match
        # inputs
        incr1 = ord(left.in_indices[0]) - ord(right.out_indices[0])
        assert incr1 >= 0, f"Problem matching right tensor's {right.out_indices} to left tensor's {left.in_indices}, " \
                           f"we are assuming right tensors indices are incremented, never decremented"

        for idx in reversed(sorted(set(right.in_indices + right.out_indices + right.contracted_indices))):
            if incr1 > 0:
                right.rename_index(idx, chr(ord(idx)+incr1), 'right')

        # then increment contracted+output indices of the right to avoid interfering with left uncontracted in indices
        # actually maybe don't need to because left's contracted indices are strictly lower
        incr2 = len(right.out_indices) - len(left.in_indices)
        # assert incr2 >= 0, f"Right tensor has more output indices {right.out_indices} than left has input indices " \
        #                   f"{left.in_indices}"
        # for idx in left.contracted_indices:

        # finally, increment uncontracted input indices on the left to avoid interfering with contracted/input indices
        # left.input_indices except left's contracted input indices incremented by
        for idx in set(left.in_indices).difference(right.out_indices):
            # move forward by set(right.contracted_indices+right.in_indices) -
            offset = len(set(right.contracted_indices+right.in_indices))
            if offset > 0:
                left.rename_index(idx, chr(ord(idx)+offset), 'left')

        print('my new spec list', left._index_spec_list)
        print('right new spec list', right._index_spec_list)

        return StructuredTensor(left._index_spec_list + right._index_spec_list, left.tensors + right.tensors)

    def __mul__(self, other):
        return self.contract(other)

    @property
    def value(self):
        return torch.einsum(self._einsum_spec, *self.tensors)

    @property
    def flops(self):
        """Flops required to materialize this tensor after einsum optimization"""

        views = oe.helpers.build_views(self._einsum_spec, self.index_dim)
        path, info = oe.contract_path(self._einsum_spec, *views, optimize='dp')
        return int(info.opt_cost)

    def _print_schedule(self):
        """Prints contraction schedule obtained by einsum optimizer"""

        einsum_str = self._einsum_spec
        sizes_dict = self.index_dim

        # indices: ['ij','jk','kl','lm']
        indices = einsum_str.split('->')[0].split(',')
        output_indices = einsum_str.split('->')[1]
        # unique_inds = set(einsum_str) - {',', '-', '>'}
        # index_size = [5]*len(unique_inds)
        # sizes_dict = dict(zip(unique_inds, index_size))
        views = oe.helpers.build_views(einsum_str, sizes_dict)

        # path: contraction path in einsum optimizer format, ie, [(0,), (2,), (1, 3), (0, 2), (0, 1)]
        path, info = oe.contract_path(einsum_str, *views, optimize='dp')

        # TODO(y): replace terms with something user provided
        # terms: ['term1', 'term2', 'term3', 'term4']
        terms = [f'term{i}' for i in range(len(indices))]
        print('optimizing ', einsum_str, terms)
        print('flops: ', info.opt_cost)

        # output_subscript: ['kl']
        output_subscript = output_indices

        input_index_sets = [set(x) for x in indices]
        output_indices = set(output_subscript)

        derived_count = 0
        for i, contract_inds in enumerate(path):
            contract_inds = tuple(sorted(list(contract_inds), reverse=True))
            # print(f'contracting {contract_inds}, input {input_index_sets}, output {output_indices}')
            contract_tuple = oe_helpers.find_contraction(contract_inds, input_index_sets, output_indices)
            out_inds, input_index_sets, _, idx_contract = contract_tuple
            # print(f'idx_contract {idx_contract}, out_inds {out_inds}')

            current_input_index_sets = [indices.pop(x) for x in contract_inds]
            current_terms = [terms.pop(x) for x in contract_inds]

            # Last contraction
            if (i - len(path)) == -1:
                current_output_indices = output_subscript
                derived_term = f'derived{derived_count}'
            else:
                all_input_inds = "".join(current_input_index_sets)
                current_output_indices = "".join(sorted(out_inds, key=all_input_inds.find))
                derived_term = f'derived{derived_count}'
                derived_count += 1

            indices.append(current_output_indices)
            terms.append(derived_term)

            new_terms = []
            new_sets = []
            # for i in natsort.index_natsorted(current_terms):
            for i in natsort.index_natsorted(current_input_index_sets):
                new_terms.append(current_terms[i])
                new_sets.append(current_input_index_sets[i])
            # einsum_str = ",".join(current_input_index_sets) + "->" + current_output_indices
            #        print(f'{derived_term}=einsum({einsum_str}, {current_terms})')
            einsum_str = ",".join(new_sets) + "->" + current_output_indices
            print(f'{derived_term}=einsum({einsum_str}, {new_terms})')


class TensorContractionChain:
    """Represents contraction chain of multiple structured tensors. Keeps them in original form because one might
    call D on this

    D(dh1(f[1:]) * dh2(f[2:])) -> D(dh1(f[1:])) * dh2(f[2:]) + dh1(f[1:])*D(dh2(f[2:]))

    # supports ".flops" and ".value" fields which perform optimization/computation


    """

    children: List[StructuredTensor]

    @property
    def flops(self):
        assert 1 != 0, "not implemented"
        return 0

    @property
    def value(self):
        result = self.children[0]
        for c in self.children[1:]:
            result = result.contract(c)
        return result.value

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
