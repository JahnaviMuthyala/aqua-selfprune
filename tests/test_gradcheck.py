"""Part 1: gradient checking for every op, broadcasting, accumulation, stability."""
import numpy as np

from engine import Tensor, check_grad, softmax_cross_entropy

rng = np.random.RandomState(0)
A = lambda *s: rng.randn(*s)
away_from_zero = lambda *s: np.sign(rng.randn(*s)) * (0.2 + np.abs(rng.randn(*s)))  # keep off ReLU kink


def test_add_sub_mul():
    check_grad(lambda a, b: a + b, A(3, 4), A(3, 4))
    check_grad(lambda a, b: a - b, A(3, 4), A(3, 4))
    check_grad(lambda a, b: a * b, A(3, 4), A(3, 4))


def test_div():
    check_grad(lambda a, b: a / b, A(3, 4), np.abs(A(3, 4)) + 1.0)


def test_neg_pow_exp_log():
    check_grad(lambda a: -a, A(3, 4))
    check_grad(lambda a: a ** 3, A(3, 4))
    check_grad(lambda a: a.exp(), A(3, 4))
    check_grad(lambda a: a.log(), np.abs(A(3, 4)) + 0.5)


def test_scalar_ops_and_reflected():
    check_grad(lambda a: 2.0 - a * 3.0 + 1.0 / (a * a + 1.0), A(3, 4))


# --- broadcasting ---------------------------------------------------------------------
def test_broadcast_row_vector_add():  # the (N, D) + (D,) case from the spec
    check_grad(lambda x, b: x + b, A(5, 3), A(3))


def test_broadcast_column_and_row_mul():  # (N,1) * (1,D)
    check_grad(lambda a, b: a * b, A(5, 1), A(1, 3))


def test_broadcast_scalar_tensor():
    check_grad(lambda a, s: a * s + s, A(4, 3), A())


def test_broadcast_sub_div():
    check_grad(lambda x, b: x - b, A(5, 3), A(3))
    check_grad(lambda x, b: x / b, A(5, 3), np.abs(A(3)) + 1.0)
    check_grad(lambda x, b: b / (x * x + 1.0), A(5, 3), A(3))


def test_broadcast_grad_shape_matches_param():
    x, b = Tensor(A(5, 3), requires_grad=True), Tensor(A(3), requires_grad=True)
    (x + b).sum().backward()
    assert b.grad.shape == (3,) and np.allclose(b.grad, 5.0)


# --- linear algebra / reductions ---------------------------------------------------------
def test_matmul():
    check_grad(lambda a, b: a @ b, A(4, 5), A(5, 3))


def test_sum_mean_axes():
    for ax in (None, 0, 1, (0, 1)):
        for kd in (False, True):
            check_grad(lambda a: a.sum(axis=ax, keepdims=kd), A(3, 4))
            check_grad(lambda a: a.mean(axis=ax, keepdims=kd), A(3, 4))


# --- non-linearities -----------------------------------------------------------------
def test_relu():
    check_grad(lambda a: a.relu(), away_from_zero(4, 5))


def test_tanh_sigmoid_gelu():
    check_grad(lambda a: a.tanh(), A(4, 5))
    check_grad(lambda a: a.sigmoid(), A(4, 5))
    check_grad(lambda a: a.gelu(), A(4, 5) * 2)


def test_sigmoid_extreme_inputs_finite():
    out = Tensor(np.array([-1e4, 0.0, 1e4])).sigmoid().data
    assert np.all(np.isfinite(out)) and out[0] == 0.0 and out[2] == 1.0


# --- softmax cross-entropy ---------------------------------------------------------------
def test_softmax_ce_gradcheck():
    y = rng.randint(0, 5, size=6)
    check_grad(lambda z: softmax_cross_entropy(z, y), A(6, 5))


def test_softmax_ce_numerical_stability():
    z = Tensor(np.array([[1000.0, 0.0, -1000.0], [-1e4, 1e4, 0.0]]), requires_grad=True)
    loss = softmax_cross_entropy(z, np.array([0, 1]))
    loss.backward()
    assert np.isfinite(loss.data) and np.all(np.isfinite(z.grad))
    assert abs(float(loss.data)) < 1e-6  # both rows are confidently correct


def test_softmax_ce_matches_closed_form():
    z = A(4, 3)
    y = np.array([0, 2, 1, 1])
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    expect = -np.log(p[np.arange(4), y]).mean()
    assert np.isclose(float(softmax_cross_entropy(Tensor(z), y).data), expect)


# --- graph mechanics ---------------------------------------------------------------------
def test_gradient_accumulation_node_used_twice():
    check_grad(lambda a: a * a + a, A(3, 3))  # a feeds 3 edges
    check_grad(lambda a: (a * 2.0) * (a * 3.0) + a.tanh() * a, A(3, 3))  # diamond


def test_topological_order_deep_chain_no_recursion_error():
    x = Tensor(np.array([1.0]), requires_grad=True)
    y = x
    for _ in range(5000):  # far beyond Python's default recursion limit
        y = y * 1.0001
    y.sum().backward()
    assert np.isclose(x.grad[0], 1.0001 ** 5000)


def test_two_layer_mlp_end_to_end():
    y = rng.randint(0, 3, size=7)

    def net(x, w1, b1, w2, b2):
        return softmax_cross_entropy((x @ w1 + b1).relu() @ w2 + b2, y)

    check_grad(net, A(7, 4), A(4, 5) * 0.5, A(5), A(5, 3) * 0.5, A(3), rtol=1e-4, atol=1e-6)
