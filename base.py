"""Base types used everywhere"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Any


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

    """
    def __matmul__(self, other):
        if isinstance(other, Function):
            return FunctionComposition([self, other])
        else:
            return NotImplemented
    """


class AtomicFunction(Function):
    @property
    def d1(self):
        return self.d(1)

    @abstractmethod
    def d(self, order: int):
        pass


class LinearizedFunction(ABC):
    """This represents a function which outputs Tensor objects."""
    pass


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


class SymmetricBilinearMap(Tensor, ABC):
    """Symmetric bilinear map. Two sets of input indices with equal dimensions.
    TODO(y): enforce this somehow"""

    @abstractmethod
    def out_dims(self) -> Tuple[int]:
        pass

    @abstractmethod
    def in_dims(self) -> Tuple[int]:
        pass

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
D2 = D@D
