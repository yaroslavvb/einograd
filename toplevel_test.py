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


def test_contract():
    W0 = u.to_pytorch([[1, -2], [-3, 4]])
    U0 = u.to_pytorch([[5, -6], [-7, 8]])
    W = DenseLinear(W0)
    U = DenseLinear(U0)
    x0 = u.to_pytorch([1, 2])
    x = DenseVector(x0)
    y = DenseCovector(x0)

    u.check_equal(W * U, W0 @ U0)
    assert isinstance(W * U, LinearMap)
    u.check_equal(W * x, [-3, 5])
    assert isinstance(W * x, Vector)
    u.check_equal(y * W, [-5, 6])
    assert isinstance(y * W, Covector)
    u.check_equal(y * x, 5)
    assert isinstance(y * x, Scalar)


def _create_unit_test_a():
    W0 = u.to_pytorch([[1, -2], [-3, 4]])
    U0 = u.to_pytorch([[5, -6], [-7, 8]])
    x0 = u.to_pytorch([1, 2])

    W = LinearLayer(W0)
    U = LinearLayer(U0)
    nonlin = Relu(x0.shape[0])
    loss = LeastSquares(x0.shape[0])
    x = DenseVector(x0)
    return W0, U0, x0, x, W, nonlin, U, loss

def _create_unit_test_a_sigmoid():
    W0 = u.to_pytorch([[1, -2], [-3, 4]])
    U0 = u.to_pytorch([[5, -6], [-7, 8]])
    x0 = u.to_pytorch([1, 2])

    W = LinearLayer(W0)
    U = LinearLayer(U0)
    nonlin = Sigmoid(x0.shape[0])
    loss = LeastSquares(x0.shape[0])
    x = DenseVector(x0)
    return W0, U0, x0, x, W, nonlin, U, loss

def test_unit_test_a():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)

    # (h1, h2, h3, h4) = (W, nonlin, U, loss)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    assert type(h1) == LinearLayer
    assert type(h4) == LeastSquares

    a1 = x
    a2 = h1(a1)    # a_i gives input into i'th layer
    a3 = h2(a2)
    a4 = h3(a3)
    a5 = h4(a4)
    u.check_equal(a1, [1, 2])
    u.check_equal(a2, [-3, 5])
    u.check_equal(a3, [0, 5])
    u.check_equal(a4, [-30, 40])
    u.check_equal(a5, 1250)

    # check per-layer Jacobians
    dh1, dh2, dh3, dh4 = D(h1), D(h2), D(h3), D(h4)

    u.check_equal(dh1(a1), W0)
    u.check_equal(dh2(a2), [[0, 0], [0, 1]])
    u.check_equal(dh3(a3), [[5, -6], [-7, 8]])
    u.check_equal(dh4(a4), [-30, 40])
    u.check_equal(dh4(a4) * dh3(a3), [-430, 500])
    u.check_equal(dh4(a4) * dh3(a3) * dh2(a2), [0, 500])
    u.check_equal(dh4(a4) * dh3(a3) * dh2(a2) * dh1(a1), [-1500, 2000])

    u.reset_global_forward_flops()
    assert u.get_global_forward_flops() == 0

    result = f(x)
    u.check_equal(result, 1250)
    assert u.get_global_forward_flops() == 4
    _unused_result = f(x)
    assert u.get_global_forward_flops() == 4

    # creating new composition does not reuse cache
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    _unused_result = f(x)
    assert u.get_global_forward_flops() == 2*4

    # partial composition test
    u.reset_global_forward_flops()
    print('flops ', u.get_global_forward_flops())
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    # result = f(x)
    a2 = f[3:](x)   # input into h2
    assert u.get_global_forward_flops() == 1
    u.check_equal(a2, [-3, 5])

    a4 = f[1:](x)  #
    assert u.get_global_forward_flops() == 3
    u.check_equal(a4, [-30, 40])

    a5 = f[:](x)  #
    assert u.get_global_forward_flops() == 4
    u.check_equal(a5, 1250)

    a5 = f[0:](x)  #
    assert u.get_global_forward_flops() == 4
    u.check_equal(a5, 1250)

    #  next steps
    # call, "D" operator,


def test_sigmoid():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    a1 = x
    a2 = h1(a1)    # a_i gives input into i'th layer
    _unused_a3 = h2(a2)

    nonlin = Sigmoid(x0.shape[0])
    print('d sigmoid', D(nonlin)(a2))
    print('d2 sigmoid', D2(nonlin)(a2))
    print(D2(nonlin).order)

    u.check_close(nonlin(a2), [0.0474259, 0.993307])
    u.check_close(D(nonlin)(a2), [[0.0451767, 0], [0, 0.00664806]])
    u.check_close(D2(nonlin)(a2), [[[0.0408916, 0], [0, 0]], [[0, 0], [0, -0.00655907]]])

    assert isinstance(D2(nonlin)(a2), SymmetricBilinearMap)


def test_relu():
    f = Relu(2)
    df = f.d1  # also try D(f)
    # TODO(y): arguments to functions don't have Tensor semantics, so change type
    result = df(DenseVector([-3, 5]))
    u.check_equal(result, [[0, 0], [0, 1]])

    df = D(f)
    result = df(DenseVector([-3, 5]))
    u.check_equal(result, [[0, 0], [0, 1]])


def test_least_squares():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    a1 = x
    a2 = h1(a1)    # a_i gives input into i'th layer
    a3 = h2(a2)
    a4 = h3(a3)
    _unused_a5 = h4(a4)

    assert isinstance(D(h4)(a4), Covector)
    assert isinstance(D2(h4)(a4), QuadraticForm)
    u.check_equal(D(h4)(a4), a4)
    u.check_equal(D2(h4)(a4), torch.eye(2))


def test_contraction():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)

    # (h1, h2, h3, h4) = (W, nonlin, U, loss)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    assert type(h1) == LinearLayer
    assert type(h4) == LeastSquares

    a1 = x
    a2 = h1(a1)    # a_i gives input into i'th layer
    a3 = h2(a2)
    a4 = h3(a3)
    a5 = h4(a4)

    f(a1)   # run once to save activations

    u.check_equal(a1, [1, 2])
    u.check_equal(a2, [-3, 5])
    u.check_equal(a3, [0, 5])
    u.check_equal(a4, [-30, 40])
    u.check_equal(a5, 1250)

    # check per-layer Jacobians
    dh1, dh2, dh3, dh4 = D(h1), D(h2), D(h3), D(h4)

    _unused_deriv = dh1(f[1:]) * dh2(f[2:]) * dh3(f[3:])

    # TODO: change D functions to produce "structured tensor" objects
    # print(deriv.flops)  # prints the flop count
    # print(deriv.value)   # prints the value

def test_structured_tensor():
    d = 10
    x00 = torch.ones((d, d))
    y00 = torch.ones((d, d))
    z00 = 2*torch.ones((d, d))

    a = StructuredTensor(['a|b', 'b|c'], [x00, y00])
    u.check_equal(a, x00 @ y00)
    assert a.flops == 2*d**3

    x = StructuredTensor(['i|j', 'j|k', 'k|l'], [x00, y00, z00])
    u.check_equal(x, x00 @ y00 @ z00)

    x = StructuredTensor(['a|b'], [x00])
    y = StructuredTensor(['a|b'], [y00])
    z = StructuredTensor(['a|b'], [z00])

    # sanity basic FLOP counts from
    # https://www.dropbox.com/s/47jxfhkb5g9nwvb/einograd-flops-basic.pdf?dl=0

    x.contract(y)
    xyz = x*y*z
    assert xyz.flops == 4*d**3
    u.check_equal(xyz, x00 @ y00 @ z00)

    x00 = torch.ones((d,))
    ma0 = 2*torch.ones(d, d)
    col = StructuredTensor.from_dense_vector(x00, 'col')
    row = StructuredTensor.from_dense_covector(x00, 'row')
    mat = StructuredTensor.from_dense_matrix(ma0, 'mat')

    assert (row * mat * mat * mat).flops == 600  # reverse mode
    assert (mat * mat * mat * col).flops == 600  # forward mode

    #    assert (mat * mat * col * row * mat * mat).flops == 1000 # mixed mode
    assert (col * row).flops == d*d                          # outer product

    u.check_equal(row * mat * mat * mat,
                  x00 @ ma0 @ ma0 @ ma0)
    u.check_equal(mat * mat * mat * col,
                  ma0 @ ma0 @ ma0 @ x00)
    colmat000 = torch.outer(x00, x00)
    u.check_equal(mat * mat * col * row * mat * mat,
                  ma0 @ ma0 @ colmat000 @ ma0 @ ma0)

    diag = StructuredTensor.from_diag_matrix(3 * x00, 'diag')
    dia0 = diag.value
    print(dia0)

    assert (row * mat * diag * mat).flops == 410             # structured reverse mode

    print()
    u.check_equal(row * mat * diag * mat,
                  x00 @ ma0 @ dia0 @ ma0)

    # 3 x 2 grid example from "3 decompositions" section of
    # https://notability.com/n/wNU5UXNGENsmRBzMFDSJQ
    d = 2
    rank2 = torch.ones((d, d))
    rank3 = torch.ones((d, d, d))
    A = StructuredTensor.from_dense_covector(rank2, idx='ij', tag='A')
    B = StructuredTensor.from_dense_linearmap(rank3, idx='i|ml', tag='B')
    # C = StructuredTensor.from_dense_linearmap(rank2, idx='l|o', tag='C')
    # D = StructuredTensor.from_dense_linearmap(rank2, idx='j|k', tag='D')
    # E = StructuredTensor.from_dense_linearmap(rank3, idx='km|n', tag='E')
    # F = StructuredTensor.from_dense_vector(rank2, idx='no', tag='F')
    print(A * B)
    # K = A * B
    # K = A * B * C * D * E * F
    # disable some error checks
    # gl.ALLOW_PARTIAL_CONTRACTIONS = True
    # gl.ALLOW_UNSORTED_INDICES = True
    # K = A * B * C * D
    # TODO(y): non-determinism (probably because using set)
    #    print(K.value)
    #    print(K.flops)

def test_contractible_tensor():
    d = 10
    x00 = torch.ones((d, d))
    y00 = torch.ones((d, d))
    z00 = 2*torch.ones((d, d))

    a = ContractibleTensor(['a|b', 'b|c'], [x00, y00])
    u.check_equal(a, x00 @ y00)
    assert a.flops == 2*d**3

    x = ContractibleTensor(['i|j', 'j|k', 'k|l'], [x00, y00, z00])
    u.check_equal(x, x00 @ y00 @ z00)

    x = ContractibleTensor(['a|b'], [x00])
    y = ContractibleTensor(['a|b'], [y00])
    z = ContractibleTensor(['a|b'], [z00])

    # sanity basic FLOP counts from
    # https://www.dropbox.com/s/47jxfhkb5g9nwvb/einograd-flops-basic.pdf?dl=0

    a = ContractibleTensor(['a|b'], [x00])
    b = ContractibleTensor(['a|b'], [x00])
    print('----')

    print(a)
    print('----')
    sys.exit()
    c = a * b
    assert c.index_spec_list == ['a|b','b|c']

    print('----')
    print(c)
    print('----')
    sys.exit()

    x.contract(y)
    xyz = x*y*z
    assert xyz.flops == 4*d**3
    u.check_equal(xyz, x00 @ y00 @ z00)

    x00 = torch.ones((d,))
    ma0 = 2*torch.ones(d, d)
    col = StructuredTensor.from_dense_vector(x00, 'col')
    row = StructuredTensor.from_dense_covector(x00, 'row')
    mat = StructuredTensor.from_dense_matrix(ma0, 'mat')

    assert (row * mat * mat * mat).flops == 600  # reverse mode
    assert (mat * mat * mat * col).flops == 600  # forward mode

    #    assert (mat * mat * col * row * mat * mat).flops == 1000 # mixed mode
    assert (col * row).flops == d*d                          # outer product

    u.check_equal(row * mat * mat * mat,
                  x00 @ ma0 @ ma0 @ ma0)
    u.check_equal(mat * mat * mat * col,
                  ma0 @ ma0 @ ma0 @ x00)
    colmat000 = torch.outer(x00, x00)
    u.check_equal(mat * mat * col * row * mat * mat,
                  ma0 @ ma0 @ colmat000 @ ma0 @ ma0)

    diag = StructuredTensor.from_diag_matrix(3 * x00, 'diag')
    dia0 = diag.value
    print(dia0)

    assert (row * mat * diag * mat).flops == 410             # structured reverse mode

    print()
    u.check_equal(row * mat * diag * mat,
                  x00 @ ma0 @ dia0 @ ma0)

    # 3 x 2 grid example from "3 decompositions" section of
    # https://notability.com/n/wNU5UXNGENsmRBzMFDSJQ
    d = 2
    rank2 = torch.ones((d, d))
    rank3 = torch.ones((d, d, d))
    A = StructuredTensor.from_dense_covector(rank2, idx='ij', tag='A')
    B = StructuredTensor.from_dense_linearmap(rank3, idx='i|ml', tag='B')
    # C = StructuredTensor.from_dense_linearmap(rank2, idx='l|o', tag='C')
    # D = StructuredTensor.from_dense_linearmap(rank2, idx='j|k', tag='D')
    # E = StructuredTensor.from_dense_linearmap(rank3, idx='km|n', tag='E')
    # F = StructuredTensor.from_dense_vector(rank2, idx='no', tag='F')
    print(A * B)
    # K = A * B
    # K = A * B * C * D * E * F
    # disable some error checks
    # gl.ALLOW_PARTIAL_CONTRACTIONS = True
    # gl.ALLOW_UNSORTED_INDICES = True
    # K = A * B * C * D
    # TODO(y): non-determinism (probably because using set)
    #    print(K.value)
    #    print(K.flops)


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


def test_derivatives():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (W, nonlin, U, loss) = (h1, h2, h3, h4)

    # sum rule
    expr1 = D(W+U)
    expr2 = D(W) + D(U)
    u.check_equal(expr1(x), expr2(x))

    # product rule
    expr1 = D(W*U)
    expr2 = D(W)*U + D(U)*W
    u.check_equal(expr1(x), expr2(x))

    # chain rule
    expr1 = D(W@U)
    expr2 = (D(W)@U)*D(U)
    u.check_equal(expr1(x), expr2(x))

    # chain rule with memoization
    gl.function_call_count = 0
    chain = MemoizedFunctionComposition(W, U)  # TODO(y): replace with W @ U
    expr1 = D(chain)
    expr2 = (D(chain[0])@chain[1:]) @ D(chain[1])
    u.check_equal(expr1(x), expr2(x))
    assert gl.function_call_count == 2  # value of U(x) is requested twice, but computed once

def test_present():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)

    # (h1, h2, h3, h4) = (W, nonlin, U, loss)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    hess = (D @ D)(f)
    u.check_equal(hess(x0), [[900., -1200.], [-1200., 1600.]])
    hvp = hess(x0) * x0
    u.check_equal(hvp, [-1500., 2000.])
    print(hvp.backward_flops)
    print(hvp)

    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a_sigmoid()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    hess = (D @ D)(f)
    u.check_equal(hess(x0), [[-8.62673, 13.5831], [13.5831, -22.3067]])
    hvp = hess(x0) * x0
    u.check_equal(hvp, [18.5394, -31.0303])
    print(hvp.backward_flops)
    print(hvp)


def run_all():
    test_contract()
    test_dense()
    test_unit_test_a()
    test_sigmoid()
    test_relu()
    test_least_squares()
    test_contraction()
    test_structured_tensor()
    test_contractible_tensor()
    #test_derivatives()

if __name__ == '__main__':
    run_all()
    sys.exit()
    # noinspection PyTypeChecker,PyUnreachableCode
    u.run_all_tests(sys.modules[__name__])
