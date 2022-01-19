"""Base types used everywhere"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Any

from attrdict import AttrDict

gl = AttrDict({'DEBUG': True, 'device': 'cpu', 'PURE_TENSOR_NETWORKS': False,
               'tensor_count': 0, 'ALLOW_PARTIAL_CONTRACTIONS': False,
               'ALLOW_UNSORTED_INDICES': False})


##################################################
# Tensors
##################################################

class Tensor(ABC):
    """
    Tensor object, corresponding to multilinear function, is characterized by a set of indices
    and their dimensions.

    Additionally, indices are split into "input" and "output" indices.

    For multilinear functions (like Hessian and higher derivatives), input indices are partitioned
    corresponding to multiple inputs. IE,
    ie, {out:ab in:{cd,ef}}

    Consider specifying symmetric structure explicitly
    ie, {out:ab, in:{cd: 2}}

    """

    @abstractmethod
    def in_dims(self):
        pass

    @abstractmethod
    def out_dims(self):
        pass


class ZeroTensor(Tensor):
    """Arbitrary shape tensor of zeros"""

    def in_dims(self):
        return ()

    def out_dims(self):
        return ()


zero = ZeroTensor()
Zero = ZeroTensor()  # TODO(y) remove one of the zeros


class Scalar(Tensor, ABC):
    """Scalar, empty output and input indices"""

    def out_dims(self):
        return ()

    def in_dims(self):
        return ()


class Vector(Tensor, ABC):
    """pure contravariant Tensor, upper (output) indices only"""

    @abstractmethod
    def out_dims(self) -> Tuple[int]:
        pass


class Covector(Tensor, ABC):
    """pure covariant Tensor, lower (input) indices only"""

    @abstractmethod
    def in_dims(self) -> Tuple[int]:
        pass


class LinearMap(Tensor, ABC):
    """mixed Tensor, one set of upper and one set of lower"""

    @abstractmethod
    def out_dims(self) -> Tuple[int]:
        pass

    @abstractmethod
    def in_dims(self) -> Tuple[int]:
        pass


class QuadraticForm(Tensor, ABC):
    # this must return ()
    @abstractmethod
    def out_dims(self) -> Tuple[int]:
        pass

    @abstractmethod
    def in_dims(self) -> Tuple[int]:
        pass


class SymmetricBilinearMap(Tensor, ABC):
    """Symmetric bilinear map. Two sets of input indices with equal dimensions.
    TODO(y): enforce this somehow"""

    @abstractmethod
    def out_dims(self) -> Tuple[int]:
        pass

    @abstractmethod
    def in_dims(self) -> Tuple[int]:
        pass


class IdentityLinearMap(Tensor):
    """tensor representing identity linear map"""

    def in_dims(self):
        return ()

    def out_dims(self):
        return ()


##################################################
# Functions
##################################################


class Function(ABC):
    """Differentiable function"""

    @abstractmethod
    def __call__(self, t: 'Tensor'):
        pass

    @abstractmethod
    def in_dims(self):
        """Input (lower) dimensions"""
        pass

    @abstractmethod
    def out_dims(self):
        """Output (upper) dimensions"""
        pass


class FunctionSharedImpl(ABC):
    # addition
    def __add__(self, other: Function):
        assert isinstance(other, Function)
        if isinstance(self, FunctionAddition):
            return FunctionAddition(self.children + [other])
        else:
            return FunctionAddition([self, other])

    # contraction
    def __mul__(self, other: Function):
        assert isinstance(other, Function)
        if isinstance(self, FunctionContraction):
            return FunctionContraction(self.children + [other])
        else:
            return FunctionContraction([self, other])

    # composition
    def __matmul__(self, other: Function):
        assert isinstance(other, Function)

        if isinstance(self, FunctionComposition):
            return FunctionComposition(self.children + [other])
        else:
            return FunctionComposition([self, other])


class CompositeFunction(Function, FunctionSharedImpl, ABC):
    """Function defined as a combination of AtomicFunction objects using +, *, @"""

    children: List[Function]

    # TODO(y) drop dimensions? These are are only needed at tensor level
    def out_dims(self):
        pass

    def in_dims(self):
        pass


class FunctionAddition(CompositeFunction):
    def __init__(self, children: List['Function']):
        # Must have two children. Otherwise, it's harder to tell if two functions are the same
        # ie, FunctionAddition([f]) == FunctionContraction([f])
        assert len(children) > 2
        self.children = children

    def __call__(self, t: 'Tensor'):
        result = self.children[0]
        for c in self.children[1:]:
            result = result + c(t)
        return result


class FunctionContraction(CompositeFunction):
    def __init__(self, children: List['Function']):
        assert len(children) > 2
        self.children = children

    def __call__(self, t: 'Tensor'):
        result = self.children[0]
        for c in self.children[1:]:
            result = result * c(t)
        return result


class FunctionComposition(CompositeFunction):
    def __init__(self, children: List['Function']):
        assert len(children) > 2
        self.children = children

    def __call__(self, t: 'Tensor'):
        result = self.children[-1]
        for c in self.children[:-1]:
            result = c(result)
        return result


class AtomicFunction(Function, FunctionSharedImpl):
    @property
    def d1(self):
        return self.d(1)

    @abstractmethod
    def d(self, order: int):
        pass


class LinearizedFunction(ABC):
    """This represents a function which outputs Tensor objects."""
    pass


##################################################
# Operators
##################################################

class Operator(ABC):
    other: 'Function'

    def __call__(self, other):
        pass

    # def __matmul__(self, other):
    #    return OperatorComposition([self, other])


# We only have 1 Operator for now. Later could add Trace operator, but don't need to separate
# composition logic for now
class OperatorComposition:
    children: List[Operator]

    def __init__(self, children):
        self.children = children

    # composition operation
    def __matmul__(self, other):
        if isinstance(other, Operator):
            return OperatorComposition([self, other])
        else:
            return NotImplemented


class D_(Operator):
    """Differentiation of arbitrary order. IE D_(1) for derivative, D_(2) for Hessian"""
    order: int

    def __init__(self, order=1):
        assert order >= 1
        self.order = order

    # composition operation
    def __matmul__(self, other):
        if isinstance(other, D_):
            return D_(self.order + other.order)
        elif isinstance(other, Operator):  # TODO(y): maybe reuse Operator logic here
            return OperatorComposition([self, other])
        else:
            return NotImplemented

    # TODO(y): change to AtomicFunction return type
    def __call__(self, other: AtomicFunction) -> Callable[[], Any]:
        assert isinstance(other, AtomicFunction), "Can only differentiate atomic functions"
        return other.d(self.order)
        # implement derivative rules here


D = D_(order=1)
D2 = D @ D
