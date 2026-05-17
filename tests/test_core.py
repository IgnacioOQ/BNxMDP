import numpy as np
import pytest

from bn_mdp.core import Variable, Assignment, Factor


# ---------------------------------------------------------------------------
# Variable
# ---------------------------------------------------------------------------

def test_variable_basic():
    v = Variable("X", [0, 1])
    assert v.name == "X"
    assert v.domain == (0, 1)


def test_variable_frozen():
    v = Variable("X", [0, 1])
    with pytest.raises(Exception):
        v.name = "Y"


def test_variable_min_domain():
    with pytest.raises(ValueError):
        Variable("X", [0])


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def test_assignment_valid():
    X = Variable("X", [0, 1])
    a = Assignment()
    a[X] = 1
    assert a[X] == 1


def test_assignment_invalid_value():
    X = Variable("X", [0, 1])
    a = Assignment()
    with pytest.raises(ValueError):
        a[X] = 2


def test_assignment_from_dict():
    X = Variable("X", [0, 1])
    Y = Variable("Y", ["a", "b"])
    a = Assignment.from_dict({X: 0, Y: "a"})
    assert a[X] == 0
    assert a[Y] == "a"


def test_assignment_restrict():
    X = Variable("X", [0, 1])
    Y = Variable("Y", ["a", "b"])
    a = Assignment.from_dict({X: 0, Y: "a"})
    r = a.restrict([X])
    assert X in r
    assert Y not in r


# ---------------------------------------------------------------------------
# Factor
# ---------------------------------------------------------------------------

def make_xy():
    X = Variable("X", [0, 1])
    Y = Variable("Y", [0, 1])
    # P(Y | X): shape (2, 2), rows=X values, cols=Y values
    table = np.array([[0.8, 0.2], [0.3, 0.7]])
    f = Factor([X, Y], table)
    return X, Y, f


def test_factor_shape():
    X, Y, f = make_xy()
    assert f.table.shape == (2, 2)


def test_factor_wrong_shape():
    X = Variable("X", [0, 1])
    Y = Variable("Y", [0, 1])
    with pytest.raises(ValueError):
        Factor([X, Y], np.ones((3, 2)))


def test_factor_get():
    X, Y, f = make_xy()
    assert f.get({X: 0, Y: 0}) == pytest.approx(0.8)
    assert f.get({X: 1, Y: 1}) == pytest.approx(0.7)


def test_factor_marginalize():
    X, Y, f = make_xy()
    # Marginalise out Y: result should be all-ones (rows sum to 1)
    mX = f.marginalize(Y)
    assert mX.scope == (X,)
    np.testing.assert_allclose(mX.table, [1.0, 1.0])


def test_factor_reduce():
    X, Y, f = make_xy()
    r = f.reduce(X, 0)
    assert r.scope == (Y,)
    np.testing.assert_allclose(r.table, [0.8, 0.2])


def test_factor_product():
    X = Variable("X", [0, 1])
    Y = Variable("Y", [0, 1])
    fX = Factor([X], np.array([0.4, 0.6]))
    # P(Y | X)
    fYX = Factor([X, Y], np.array([[0.8, 0.2], [0.3, 0.7]]))
    joint = fX.product(fYX)
    assert set(joint.scope) == {X, Y}
    # P(X=0, Y=1) = P(X=0) * P(Y=1|X=0) = 0.4 * 0.2 = 0.08
    assert joint.get({X: 0, Y: 1}) == pytest.approx(0.08)


def test_factor_normalize():
    X = Variable("X", [0, 1])
    f = Factor([X], np.array([2.0, 3.0]))
    fn = f.normalize()
    np.testing.assert_allclose(fn.table, [0.4, 0.6])
