import sys

import util as u
from layers import *


def test_test():
    print('passed')


class LinearFunction(Function, Tensor, ABC):
    """Linear function with mixed Tensor, both upper and lower indices"""

    @abstractmethod
    def out_dims(self) -> Tuple[int]:
        pass

    @abstractmethod
    def in_dims(self) -> Tuple[int]:
        pass


def test_dense():
    W0 = u.to_pytorch([[1, -2], [-3, 4]])
    # U0 = u.to_pytorch([[5, -6], [-7, 8]])
    x0 = u.to_pytorch([1, 2])

    W = LinearLayer(W0)
    x = DenseVector(x0)
    u.check_equal(W(x).value, W0 @ x0)

    dW = D(W)  # derivative of linear layer
    print(dW(zero) * x)  # get
    u.check_equal(dW(zero) * x, W0 @ x0)


def test_unit_test_a():
    W0 = u.to_pytorch([[1, -2], [-3, 4]])
    U0 = u.to_pytorch([[5, -6], [-7, 8]])
    x0 = u.to_pytorch([1, 2])

    W = LinearLayer(W0)
    U = LinearLayer(U0)
    nonlin = Relu(x0.shape[0])
    loss = CrossEntropy(x0.shape[0])
    x = DenseVector(x0)

    (h1, h2, h3, h4) = (W, nonlin, U, loss)
    # f = h4 @ h3 @ h2 @ h1  # => Composition (FunctionComposition)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    h = [None, f[3], f[2], f[1], f[0]]  # h[1] shorthand for h1, linked to a parent Composition
    assert type(h[1]) == LinearLayer
    assert type(h[4]) == CrossEntropy

    a1 = x
    a2 = h1(a1)    # ai gives input into i'th layer
    a3 = h2(a2)
    a4 = h3(a3)
    a5 = h4(a4)
    u.check_equal(a1, [1, 2])
    u.check_equal(a2, [-3, 5])
    #u.check_equal(a3, [0, 5])
    #u.check_equal(a4, [-30, 40])
    #u.check_equal(a5, 1250)

    # D(f)   # this is numerically equivalent to D(U) @ W * D(W)
    # slow = D(U) @ W * D(W)
    # fast = D(f) @ f[1:] * D(f[1])

    # print(slow(x0).forward_flops)  # high
    # print(fast(x0).forward_flops)   # low


def test_einsum():
    W0 = u.to_pytorch([[1, -2], [-3, 4]])
    U0 = u.to_pytorch([[5, -6], [-7, 8]])
    x0 = u.to_pytorch([1, 2])

    W = LinearLayer(W0)
    U = LinearLayer(U0)
    nonlin = Relu(x0.shape[0])
    dnonlin = D(nonlin)

    # W.lazy_call(x0)  # saved x0
    # dnonlin = DSigmoid(x0.shape[0])
    # dnonlin.lazy_call(x0)


def test_relu():
    f = Relu(2)
    df = f.d  # also try D(f)
    # TODO(y): arguments to functions don't have Tensor semantics, so change type
    result = df(DenseVector([-3, 5]))
    u.check_equal(result, [[0, 0], [0, 1]])

    df = D(f)
    result = df(DenseVector([-3, 5]))
    u.check_equal(result, [[0, 0], [0, 1]])


"""
def test_unit_test_A():

    W = Linear([[1, -2], [-3, 4]])
    U = Linear([[5, -6], [-7, 8]])
    x0 = make_vector([1, 2])
    nonlin = make_sigmoid(x0)
    loss = make_xent(x0)   # x0 used for shape inference

    (h1, h2, h3, h4) = (W, nonlin, U, loss)
    f = h4 @ h3 @ h2 @ h1  # => Composition (FunctionComposition)
    h = [None, f[3], f[2], f[1], f[0]]  # h[1] shorthand for h1, linked to a parent Composition
    assert type(h[1]) == Linear

    D(f)   # this is numerically equivalent to D(U) @ W * D(W)
    slow = D(U) @ W * D(W)
    fast = D(f) @ f[1:] * D(f[1])

    print(slow(x0).forward_flops)  # high
    print(fast(x0).forward_flops)   # low

    D(f)    # LazyDLinear # represents derivative of linear layer
    nonlin = make_sigmoid(x0)
    D(nonlin)   # LazyDRelu #

def old_test_unit_test_A():
    # W = DenseMatrix([[1, -2], [-3, 4]])
    # U = DenseMatrix([[5, -6], [-7, 8]])
    # print(W._indices())
    # x = DenseVector([1, 2])
    # loss = CrossEntropy()
    # layers = [W, nonlin, U]
    W = make_linear([[1, -2], [-3, 4]])
    U = make_linear([[5, -6], [-7, 8]])
    x0 = make_vector([1, 2])
    nonlin = make_relu(x0)
    loss = make_xent(x0)   # x0 used for shape inference

    (h1, h2, h3, h4) = (W, nonlin, U, loss)
    h = [None, W, nonlin, U, loss]  # h[1] shorthand for h1

    # TODO(y): add call count and make sure memoization is happening

    # go through Mechanics of converting this network to einsum notation
    f = h4 @ h3 @ h2 @ h1  # => Composition (FunctionComposition)
    df = D(f)  # Linear Function

    # chain rule with no memoization
    assert (D(h4) @ h3 @ h2 @ h1) * D(h3) @ h2 @ h1 * D(h2) @ h1 * D(h1) == df

    # chain rule with memoization
    D(h4)  # @ f[1:] # @ D(h3) @ f[2:] @ D(h2) @ f[3:] @ D(h1)
"""

"""
def test_present0():
    # reverse mode

    from einograd import jacobian, forward, to_expression

    W = u.create_linear([[1, -2], [-3, 4]])
    U = u.create_linear([[5, -6], [-7, 8]])
    loss_func = u.LeastSquaresLoss()
    out = to_expression(nn.Sequential(W, U, U, loss_func))
    data = u.to_pytorch([1, 2])
    x = data
    d = jacobian(out, x)
    print(d.flops)  # 600 FLOPs required for optimized backward
    forward(out, x)  # runs the forward pass, saves input to every layer
    print(d.value)  # materialize the actual value

    # forward mode
    out = torch.Sequential[col, W, W, W]
    d = jacobian(out, x)
    print(d.flops)  # 600
    forward(net, x)
    print(d.value)  # materialize the actual value

    out = torch.Sequential([W, relu, W, xent])
    d = jacobian(out, x)
    print(d.flops)  # 410
    forward(net, x)
    print(d.value)  # materialize the actual value

    # cross-country mode
    net = torch.Sequential([W, W, row, col, W, W])
    d = jacobian(net, x)
    print(d.flops)  # 1000

    loss = torch.Sequential([W, relu, W, relu, xent])
    x = W.weight
    vector = Tensor(torch.ones(x.shape))
    hess = jacobian(jacobian(loss, x), x)
    hvp = hess @ vector
    print(hvp.flops)
    print(hvp.value)
"""


def test_present1():
    # TODO: simplified version with ReLU, xent instead of sigmoid, LeastSquaresLoss
    pass


if __name__ == '__main__':
    test_dense()
    test_relu()
    test_unit_test_a()
    # test_unit_test_A()
    sys.exit()
    # noinspection PyTypeChecker,PyUnreachableCode
    u.run_all_tests(sys.modules[__name__])
