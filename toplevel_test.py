import sys

import pytest

from base import *


def test_dense():
    W0 = to_pytorch([[1, -2], [-3, 4]])
    # U0 = to_pytorch([[5, -6], [-7, 8]])
    x0 = to_pytorch([1, 2])

    W = OldLinearLayer(W0)
    x = DenseVector(x0)
    check_equal(W(x).value, W0 @ x0)

    dW = D(W)  # derivative of linear layer
    print(dW(zero) * x)  # get
    check_equal(dW(zero) * x, W0 @ x0)


def test_contract():
    W0 = to_pytorch([[1, -2], [-3, 4]])
    U0 = to_pytorch([[5, -6], [-7, 8]])
    W = DenseLinear(W0)
    U = DenseLinear(U0)
    x0 = to_pytorch([1, 2])
    x = DenseVector(x0)
    y = DenseCovector(x0)

    check_equal(W * U, W0 @ U0)
    assert isinstance(W * U, LinearMap)
    check_equal(W * x, [-3, 5])
    assert isinstance(W * x, Vector)
    check_equal(y * W, [-5, 6])
    assert isinstance(y * W, Covector)
    check_equal(y * x, 5)
    assert isinstance(y * x, Scalar)


def _old_create_unit_test_a():
    W0 = to_pytorch([[1, -2], [-3, 4]])
    U0 = to_pytorch([[5, -6], [-7, 8]])
    x0 = to_pytorch([1, 2])

    W = OldLinearLayer(W0)
    U = OldLinearLayer(U0)
    nonlin = OldRelu(x0.shape[0])
    loss = OldLeastSquares(x0.shape[0])
    x = DenseVector(x0)
    return W0, U0, x0, x, W, nonlin, U, loss


def _create_unit_test_a():
    W0 = to_pytorch([[1, -2], [-3, 4]])
    U0 = to_pytorch([[5, -6], [-7, 8]])
    x0 = to_pytorch([1, 2])

    W = LinearLayer(W0)
    U = LinearLayer(U0)
    nonlin = Relu()
    loss = LeastSquares()
    x = TensorContraction.from_dense_vector(x0)
    return W0, U0, x0, x, W, nonlin, U, loss


def _create_unit_test_a_sigmoid():
    W0 = to_pytorch([[1, -2], [-3, 4]])
    U0 = to_pytorch([[5, -6], [-7, 8]])
    x0 = to_pytorch([1, 2])

    W = OldLinearLayer(W0)
    U = OldLinearLayer(U0)
    nonlin = OldSigmoid(x0.shape[0])
    loss = OldLeastSquares(x0.shape[0])
    x = DenseVector(x0)
    return W0, U0, x0, x, W, nonlin, U, loss


def test_unit_test_a():
    (W0, U0, x0, x, h1, h2, h3, h4) = _old_create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)

    # (h1, h2, h3, h4) = (W, nonlin, U, loss)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    assert type(h1) == OldLinearLayer
    assert type(h4) == OldLeastSquares

    a1 = x
    a2 = h1(a1)  # a_i gives input into i'th layer
    a3 = h2(a2)
    a4 = h3(a3)
    a5 = h4(a4)
    check_equal(a1, [1, 2])
    check_equal(a2, [-3, 5])
    check_equal(a3, [0, 5])
    check_equal(a4, [-30, 40])
    check_equal(a5, 1250)

    # check per-layer Jacobians
    dh1, dh2, dh3, dh4 = D(h1), D(h2), D(h3), D(h4)

    check_equal(dh1(a1), W0)
    check_equal(dh2(a2), [[0, 0], [0, 1]])
    check_equal(dh3(a3), [[5, -6], [-7, 8]])
    check_equal(dh4(a4), [-30, 40])
    check_equal(dh4(a4) * dh3(a3), [-430, 500])
    check_equal(dh4(a4) * dh3(a3) * dh2(a2), [0, 500])
    check_equal(dh4(a4) * dh3(a3) * dh2(a2) * dh1(a1), [-1500, 2000])

    u.reset_global_forward_flops()
    assert u.get_global_forward_flops() == 0

    result = f(x)
    check_equal(result, 1250)
    assert u.get_global_forward_flops() == 4
    _unused_result = f(x)
    assert u.get_global_forward_flops() == 4

    # creating new composition does not reuse cache
    (W0, U0, x0, x, h1, h2, h3, h4) = _old_create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    _unused_result = f(x)
    assert u.get_global_forward_flops() == 2 * 4

    # partial composition test
    u.reset_global_forward_flops()
    print('flops ', u.get_global_forward_flops())
    (W0, U0, x0, x, h1, h2, h3, h4) = _old_create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    # result = f(x)
    a2 = f[3:](x)  # input into h2
    assert u.get_global_forward_flops() == 1
    check_equal(a2, [-3, 5])

    a4 = f[1:](x)  #
    assert u.get_global_forward_flops() == 3
    check_equal(a4, [-30, 40])

    a5 = f[:](x)  #
    assert u.get_global_forward_flops() == 4
    check_equal(a5, 1250)

    a5 = f[0:](x)  #
    assert u.get_global_forward_flops() == 4
    check_equal(a5, 1250)

    #  next steps
    # call, "D" operator,


def test_sigmoid():
    (W0, U0, x0, x, h1, h2, h3, h4) = _old_create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    a1 = x
    a2 = h1(a1)  # a_i gives input into i'th layer
    _unused_a3 = h2(a2)

    nonlin = OldSigmoid(x0.shape[0])
    print('d sigmoid', D(nonlin)(a2))
    print('d2 sigmoid', D2(nonlin)(a2))
    print(D2(nonlin).order)

    check_close(nonlin(a2), [0.0474259, 0.993307])
    check_close(D(nonlin)(a2), [[0.0451767, 0], [0, 0.00664806]])
    check_close(D2(nonlin)(a2), [[[0.0408916, 0], [0, 0]], [[0, 0], [0, -0.00655907]]])

    assert isinstance(D2(nonlin)(a2), SymmetricBilinearMap)


def test_relu():
    f = OldRelu(2)
    df = f.d1  # also try D(f)
    # TODO(y): arguments to functions don't have Tensor semantics, so change type
    result = df(DenseVector([-3, 5]))
    check_equal(result, [[0, 0], [0, 1]])

    df = D(f)
    result = df(DenseVector([-3, 5]))
    check_equal(result, [[0, 0], [0, 1]])


def test_least_squares():
    (W0, U0, x0, x, h1, h2, h3, h4) = _old_create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    a1 = x
    a2 = h1(a1)  # a_i gives input into i'th layer
    a3 = h2(a2)
    a4 = h3(a3)
    _unused_a5 = h4(a4)

    assert isinstance(D(h4)(a4), Covector)
    assert isinstance(D2(h4)(a4), QuadraticForm)
    check_equal(D(h4)(a4), a4)
    check_equal(D2(h4)(a4), torch.eye(2))


def test_contraction():
    (W0, U0, x0, x, h1, h2, h3, h4) = _old_create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)

    # (h1, h2, h3, h4) = (W, nonlin, U, loss)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    assert type(h1) == OldLinearLayer
    assert type(h4) == OldLeastSquares

    a1 = x
    a2 = h1(a1)  # a_i gives input into i'th layer
    a3 = h2(a2)
    a4 = h3(a3)
    a5 = h4(a4)

    f(a1)  # run once to save activations

    check_equal(a1, [1, 2])
    check_equal(a2, [-3, 5])
    check_equal(a3, [0, 5])
    check_equal(a4, [-30, 40])
    check_equal(a5, 1250)

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
    z00 = 2 * torch.ones((d, d))

    a = OldStructuredTensor(['a|b', 'b|c'], [x00, y00])
    check_equal(a, x00 @ y00)
    assert a.flops == 2 * d ** 3

    x = OldStructuredTensor(['i|j', 'j|k', 'k|l'], [x00, y00, z00])
    check_equal(x, x00 @ y00 @ z00)

    x = OldStructuredTensor(['a|b'], [x00])
    y = OldStructuredTensor(['a|b'], [y00])
    z = OldStructuredTensor(['a|b'], [z00])

    # sanity basic FLOP counts from
    # https://www.dropbox.com/s/47jxfhkb5g9nwvb/einograd-flops-basic.pdf?dl=0

    x.contract(y)
    xyz = x * y * z
    assert xyz.flops == 4 * d ** 3
    check_equal(xyz, x00 @ y00 @ z00)

    x00 = torch.ones((d,))
    ma0 = 2 * torch.ones(d, d)
    col = OldStructuredTensor.from_dense_vector(x00, 'col')
    row = OldStructuredTensor.from_dense_covector(x00, 'row')
    mat = OldStructuredTensor.from_dense_matrix(ma0, 'mat')

    assert (row * mat * mat * mat).flops == 600  # reverse mode
    assert (mat * mat * mat * col).flops == 600  # forward mode

    #    assert (mat * mat * col * row * mat * mat).flops == 1000 # mixed mode
    assert (col * row).flops == d * d  # outer product

    check_equal(row * mat * mat * mat,
                  x00 @ ma0 @ ma0 @ ma0)
    check_equal(mat * mat * mat * col,
                  ma0 @ ma0 @ ma0 @ x00)
    colmat000 = torch.outer(x00, x00)
    check_equal(mat * mat * col * row * mat * mat,
                  ma0 @ ma0 @ colmat000 @ ma0 @ ma0)

    diag = OldStructuredTensor.from_diag_matrix(3 * x00, 'diag')
    dia0 = diag.value
    print(dia0)

    assert (row * mat * diag * mat).flops == 410  # structured reverse mode

    print()
    check_equal(row * mat * diag * mat,
                  x00 @ ma0 @ dia0 @ ma0)

    # 3 x 2 grid example from "3 decompositions" section of
    # https://notability.com/n/wNU5UXNGENsmRBzMFDSJQ
    d = 2
    rank2 = torch.ones((d, d))
    rank3 = torch.ones((d, d, d))
    A = OldStructuredTensor.from_dense_covector(rank2, idx='ij', tag='A')
    B = OldStructuredTensor.from_dense_linearmap(rank3, idx='i|ml', tag='B')
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

def test_contractible_tensor2():
    d = 10
    x00 = torch.ones((d, d))
    y00 = torch.ones((d, d))
    z00 = 2 * torch.ones((d, d))

    diag = TensorContraction.from_diag_matrix(3 * torch.ones((3,)), label='diag')
    assert diag.out_idx == diag.in_idx
    assert len(diag.out_idx) == 1
    dia0 = diag.value
    print(dia0)

    a = TensorContraction([('a|b', x00), ('b|c', y00)])
    check_equal(a.value, x00 @ y00)
    assert a.flops == 2 * d ** 3

    x = TensorContraction.__legacy_init__(['i|j', 'j|k', 'k|l'], [x00, y00, z00])
    check_equal(x, x00 @ y00 @ z00)

    x = TensorContraction.__legacy_init__(['a|b'], [x00])
    y = TensorContraction.__legacy_init__(['a|b'], [y00])
    z = TensorContraction.__legacy_init__(['a|b'], [z00])

    # sanity basic FLOP counts from
    # https://www.dropbox.com/s/47jxfhkb5g9nwvb/einograd-flops-basic.pdf?dl=0

    a = TensorContraction.__legacy_init__(['a|b'], [x00])
    b = TensorContraction.__legacy_init__(['a|b'], [x00])
    print(a)
    c = a * b
    assert c.children_specs == ['a|b', 'b|c']
    check_equal(c, x00 @ x00)

    d2 = 2
    rank1 = torch.ones((d2,))
    rank2 = torch.ones((d2, d2))
    rank3 = torch.ones((d2, d2, d2))
    rank4 = torch.ones((d2, d2, d2, d2))
    rank6 = torch.ones((d2, d2, d2, d2, d2, d2))
    a = TensorContraction.__legacy_init__(['a|bc', 'bc|de'], [rank3, rank4], label='a')
    b = TensorContraction.__legacy_init__(['ab|c', 'c|d'], [rank3, rank2], label='b')
    c = a * b
    assert c.children_specs == ['a|bc', 'bc|de', 'de|f', 'f|g']

    a = TensorContraction.__legacy_init__(['a|bc', 'bc|defg'], [rank3, rank6], label='a')
    b = TensorContraction.__legacy_init__(['ab|c', 'c|d'], [rank3, rank2], label='b')
    c = a * b
    assert c.children_specs == ['a|bc', 'bc|defg', 'de|h', 'h|i']

    a = TensorContraction.__legacy_init__(['a|'], [rank1])
    b = TensorContraction.__legacy_init__(['|a'], [rank1])
    c = a * b
    assert c.children_specs == ['a|', '|b']

    check_equal(c, torch.outer(rank1, rank1))

    xyz = x * y * z
    assert xyz.flops == 4 * d ** 3
    check_equal(xyz, x00 @ y00 @ z00)

    x00 = torch.ones((d,))
    ma0 = 2 * torch.ones(d, d)
    col = TensorContraction.from_dense_vector(x00, 'col')
    row = TensorContraction.from_dense_covector(x00, 'row')
    mat = TensorContraction.from_dense_matrix(ma0, 'mat')

    assert (row * mat * mat * mat).flops == 600  # reverse mode
    assert (mat * mat * mat * col).flops == 600  # forward mode

    #    assert (mat * mat * col * row * mat * mat).flops == 1000 # mixed mode
    assert (col * row).flops == d * d  # outer product

    check_equal(row * mat * mat * mat,
                  x00 @ ma0 @ ma0 @ ma0)
    check_equal(mat * mat * mat * col,
                  ma0 @ ma0 @ ma0 @ x00)
    colmat000 = torch.outer(x00, x00)
    check_equal(mat * mat * col * row * mat * mat,
                  ma0 @ ma0 @ colmat000 @ ma0 @ ma0)

    diag = TensorContraction.from_diag_matrix(3 * x00, 'diag')
    assert diag.out_idx == diag.in_idx
    assert len(diag.out_idx) == 1
    dia0 = diag.value
    print(dia0)

    d2 = 2
    rank2 = torch.ones((d2, d2))
    rank3 = torch.ones((d2, d2, d2))
    rank4 = torch.ones((d2, d2, d2, d2))

    # UnitTestB:
    A = torch.ones((2, 3, 2, 2))
    B = torch.ones((2,))
    C = torch.ones((2, 2, 2, 4))

    D = TensorContraction([('ab|cd', A, 'A'), ('c|c', B, 'B'), ('cd|ef', C, 'C')])
    assert D.ricci_str == 'ab|cd,c|c,cd|ef->ab|ef'

    a = TensorContraction.__legacy_init__(['a|bc', 'bc|de'], [rank3, rank4], label='a')
    b = TensorContraction.__legacy_init__(['ab|c', 'c|d'], [rank3, rank2], label='b')
    c = a * b

    check_equal(row * diag, x00 @ dia0)
    check_equal(row * mat * diag, x00 @ ma0 @ dia0)

    result = row*mat
    print(result.ricci_str)
    assert result.ricci_str == '|a,a|b->|b'
    assert (row*mat*diag).ricci_str == '|a,a|b,b|b->|b'
    assert (row * mat * diag).ricci_str == '|a,a|b,b|b->|b'
    assert (row * mat * diag * mat).flops == 410  # structured reverse mode

    print()
    check_equal(row * mat * diag * mat,
                  x00 @ ma0 @ dia0 @ ma0)

    # 2x3 grid example from "3 decompositions" section of
    # https://notability.com/n/wNU5UXNGENsmRBzMFDSJQ
    d = 2
    rank2 = torch.ones((d, d))
    rank3 = torch.ones((d, d, d))

    d2 = 2
    rank1 = torch.ones((d2,))
    rank2 = torch.ones((d2, d2))
    rank3 = torch.ones((d2, d2, d2))
    rank4 = torch.ones((d2, d2, d2, d2))
    rank6 = torch.ones((d2, d2, d2, d2, d2, d2))

    A = ('|ij', rank2, 'A')
    B = ('i|lm', rank3, 'B')
    C = ('l|o', rank2, 'C')
    D = ('j|k', rank2, 'D')
    E = ('km|n', rank3, 'E')
    F = ('no|', rank2, 'F')
    K = TensorContraction([A, B, C, D, E, F])
    assert K.flops == 104, "Change in flop requirement detected (don't actually know if 104 is correct)"


@pytest.mark.skip(reason="this example doesn't work because our implementation currently contracts sequentially left to right only with automatic index renaming")
def test_2x3grid_mul():
    """If we want this kind of contraction to work, need to redo contraction logic to look at particular names of indices and match
    them up. Current logic just matches positions and discards original index names: all k output indices of the right are renamed to match first k input indices
    of the left."""

    d2 = 2
    rank1 = torch.ones((d2,))
    rank2 = torch.ones((d2, d2))
    rank3 = torch.ones((d2, d2, d2))
    rank4 = torch.ones((d2, d2, d2, d2))
    rank6 = torch.ones((d2, d2, d2, d2, d2, d2))

    A = TensorContraction.from_dense_covector(rank2, idx='ij', label='A')
    B = TensorContraction.from_dense_tensor('i|lm', rank3, 'B')
    C = TensorContraction.from_dense_tensor('l|o', rank2, 'C')
    D = TensorContraction.from_dense_tensor('j|k', rank2, 'D')
    E = TensorContraction.from_dense_tensor('km|n', rank3, 'E')
    F = TensorContraction.from_dense_vector(rank2, idx='no', label='F')

    print('====='*20)
    partial1 = A*B
    print('-----', partial1.ricci_str)
    assert partial1.ricci_out == '|jlm'
    partial2 = partial1 * C
    print('-----', partial2.ricci_str)
    assert partial1.ricci_out == '|jlm'
    partial3 = partial2 * D
    print('-----', partial3.ricci_str)

    # at this point error happens, D^j is contracted with B_l instead of A_j
    assert partial1.ricci_out == '|lmo'
    partial4 = partial3 * E
    assert partial4.ricci_out == '|mok'
    print('-----', partial2.ricci_str)
    partial5 = partial4 * F
    print('-----', partial2.ricci_str)

    K = A * B * C * D * E * F
    print(K.flops)
    print(K._einsum_spec)

    # Hessian-like contractions

    # this should raise an error because rank doesn't match
    with pytest.raises(Exception):
        A = TensorContraction([('a|bc', rank2, 'A')])

    assert A.ricci_str == 'a|bc'
    B = TensorContraction([('bc|defg', rank6, 'B')])
    AB = A * B
    print(AB.ricci_str)

def test_partial_contraction_UnitTestC():
    d2 = 2
    rank1 = torch.ones((d2,))
    rank2 = torch.ones((d2, d2))
    rank3 = torch.ones((d2, d2, d2))
    rank4 = torch.ones((d2, d2, d2, d2))
    rank6 = torch.ones((d2, d2, d2, d2, d2, d2))

    # this should raise an error because rank doesn't match
    with pytest.raises(Exception):
        A = TensorContraction([('a|bc', rank2, 'A')])

    A = TensorContraction([('a|bc', rank3, 'A')])
    assert A.ricci_str == 'a|bc->a|bc'
    B = TensorContraction([('bc|defg', rank6, 'B')])
    assert B.ricci_str == 'bc|defg->bc|defg'
    AB = A * B
    assert AB.ricci_str == 'a|bc,bc|defg->a|defg'
    result2 = AB * TensorContraction([('de|h', rank3, 'C'), ('h|i', rank2, 'D')])
    result1 = TensorContraction([('a|bc', rank3, 'A'), ('bc|defg', rank6, 'B'), ('de|h', rank3, 'C'), ('h|i', rank2, 'D')])
    print(result1.ricci_str)
    assert result2.ricci_out == 'a|ifg'

    # UnitTestD
    vec = TensorContraction.from_dense_vector(rank1)
    assert (result2 * vec).ricci_out == 'a|fg'
    assert (result2 * vec * vec).ricci_out == 'a|g'
    assert (result2 * vec * vec * vec).ricci_out == 'a|'


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

    W = create_linear([[1, -2], [-3, 4]])
    U = create_linear([[5, -6], [-7, 8]])
    loss_func = LeastSquaresLoss()
    out = to_expression(nn.Sequential(W, U, U, loss_func))
    data = to_pytorch([1, 2])
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


@pytest.mark.skip(reason="doesn't work yet")
def test_derivatives():
    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    (W, nonlin, U, loss) = (h1, h2, h3, h4)

    # (h1, h2, h3, h4) = (W, nonlin, U, loss)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    f_slow = FunctionComposition([h4, h3, h2, h1])
    assert type(h1) == LinearLayer
    assert type(h4) == LeastSquares

    a1 = x
    a2 = h1(a1)  # a_i gives input into i'th layer
    a3 = h2(a2)
    a4 = h3(a3)
    a5 = h4(a4)
    check_equal(a1, [1, 2])
    check_equal(a2, [-3, 5])
    check_equal(a3, [0, 5])
    check_equal(a4, [-30, 40])
    check_equal(a5, 1250)

    # check per-layer Jacobians
    dh1, dh2, dh3, dh4 = D(h1), D(h2), D(h3), D(h4)

    check_equal(dh1(a1), W0)
    check_equal(dh2(a2), [[0, 0], [0, 1]])
    check_equal(dh3(a3), [[5, -6], [-7, 8]])

    # simple hessian
    hess = D @ D
    print(hess(W)(x))
    check_equal(hess(W)(x), 0)

    check_equal(hess(loss)(x), torch.eye(2))

    first_deriv = D(loss @ W)
    second_deriv = D(first_deriv)

    print(hess(loss @ W))
    print(D(loss @ W))
    sys.exit()

    # check end-to-end derivative
    check_equal(D(f_slow)(x), [-1500, 2000])

    # sum rule
    expr1 = D(W + U)
    expr2 = D(W) + D(U)
    check_equal(expr1(x), expr2(x))
    print(D(W @ U)(x).value)

    result = hess(W)(x)
    # print(result)

    # hessian = (D@D)(loss @ W)
    # hess
    # print(hessian(a1).value)


    # chain rule with memoization
    # GLOBALS.function_call_count = 0
    #chain = MemoizedFunctionComposition([W, U])  # TODO(y): replace with W @ U
    #expr1 = D(chain)
    #expr2 = (D(chain[0]) @ chain[1:]) @ D(chain[1])
    #check_equal(expr1(x), expr2(x))
    #assert GLOBALS.function_call_count == 2  # value of U(x) is requested twice, but computed once

@pytest.mark.skip(reason="doesn't work yet")
def test_present():
    (W0, U0, x0, x, h1, h2, h3, h4) = _old_create_unit_test_a()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)

    # (h1, h2, h3, h4) = (W, nonlin, U, loss)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    hess = (D @ D)(f)
    check_equal(hess(x0), [[900., -1200.], [-1200., 1600.]])
    hvp = hess(x0) * x0
    check_equal(hvp, [-1500., 2000.])
    print(hvp.backward_flops)
    print(hvp)

    (W0, U0, x0, x, h1, h2, h3, h4) = _create_unit_test_a_sigmoid()
    (_unused_W, _unused_nonlin, _unused_U, _unused_loss) = (h1, h2, h3, h4)
    f = MemoizedFunctionComposition([h4, h3, h2, h1])
    hess = (D @ D)(f)
    check_equal(hess(x0), [[-8.62673, 13.5831], [13.5831, -22.3067]])
    hvp = hess(x0) * x0
    check_equal(hvp, [18.5394, -31.0303])
    print(hvp.backward_flops)
    print(hvp)


# Tests from "Diagonal logic"
def test_diagonal_problem():
    d = 2
    m = 20
    torch.manual_seed(1)
    row0 = torch.randint(1, m, size=(d,)).float()
    col0 = torch.randint(1, m, size=(d,)).float()
    diag0 = torch.randint(1, m, size=(d,)).float()
    ma0 = torch.randint(1, m, size=(d, d)).float()
    row = TensorContraction.from_dense_covector(row0, label='row')
    col = TensorContraction.from_dense_vector(col0, label='col')
    diag = TensorContraction.from_diag_matrix(diag0, label='diag')
    mat = TensorContraction.from_dense_matrix(ma0, label='mat')

    assert (row*diag).ricci_str == '|a,a|a->|a'
    check_equal(row*diag, row0*diag0)

    assert (diag * col).ricci_str == 'a|a,a|->a|'
    check_equal(diag * col, diag0 * col0)

    assert (row * col).ricci_str == '|a,a|->|'
    check_equal(row * col, row0 @ col0)

    # weighted dot product
    assert (row * diag * col).ricci_str == '|a,a|a,a|->|'
    check_equal(row * diag * col, (row0 * diag0 * col0).sum())

    # Hadamard product of two diagonal matrices support combining, but not direct materialization for now, need to figure out how to deal
    # with multiple diagonal matrices, only support 1
    assert (diag * diag).ricci_str == 'a|a,a|a->a|a'
    with pytest.raises(Exception):
        print((diag*diag).value)
        check_equal(diag * diag, torch.diag(diag0) @ torch.diag(diag0))

    # this case could be enabled in the future, but to reduce scope currently
    # we specialize all contractions to go in left-to-right-order
    with pytest.raises(Exception):
        assert (col * diag).ricci_str == 'a|,a|a->a|'

    check_equal(mat * diag, ma0 @ torch.diag(diag0))
    check_equal(diag * mat, torch.diag(diag0) @ ma0)

def test_diagonal_and_trace():
    A = TensorContraction([('|ab', from_numpy([[1, 2], [3, 4]]), 'A')])
    r = A.diag
    assert r.ricci_out == '|a'
    check_equal(A.diag, [1, 4])
    check_equal(A.trace, 5)

    # matrices are treated as linear forms, so no trace defined, this should raise error
    with pytest.raises(Exception):
        A = TensorContraction.from_dense_matrix(from_numpy([[1, 2], [3, 4]]))
        print(A.trace)


def run_all():
    test_derivatives()
    test_diagonal_and_trace()
    test_contractible_tensor2()
    test_partial_contraction_UnitTestC()
    test_contract()
    test_dense()
    test_unit_test_a()
    test_sigmoid()
    test_relu()
    test_least_squares()
    test_contraction()
    test_structured_tensor()
    test_contractible_tensor2()
    test_diagonal_problem()


if __name__ == '__main__':
    run_all()
    sys.exit()
    # noinspection PyTypeChecker,PyUnreachableCode
    run_all_tests(sys.modules[__name__])
