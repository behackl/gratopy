"""Tests for the randomized adjoint check of operators."""

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
import pytest

from gratopy.operator import Operator, Radon, Space

RNG = np.random.default_rng(3)
MATRIX = RNG.standard_normal((12, 5))
DOMAIN_WEIGHTS = RNG.uniform(0.5, 2.0, 5)
CODOMAIN_WEIGHTS = RNG.uniform(0.5, 2.0, 12)


class Matrix(Operator):
    """Matrix operator whose adjoint uses the given (fixed) weights."""

    def __init__(self, adjoint_weights=None, **kwargs):
        super().__init__(input_shape=(5,), output_shape=(12,), **kwargs)
        self.adjoint_weights = adjoint_weights

    def apply_to(self, argument, output=None, **kwargs):
        return MATRIX @ argument

    def apply_adjoint_to(self, argument, output=None, **kwargs):
        if self.adjoint_weights is None:
            return MATRIX.T @ argument
        w_x, w_y = self.adjoint_weights
        return (MATRIX.T @ (w_y * argument)) / w_x


def weighted_spaces():
    return Space((5,), weights=DOMAIN_WEIGHTS), Space((12,), weights=CODOMAIN_WEIGHTS)


@pytest.mark.parametrize("dtype, tolerance", [(np.float32, 1e-5), (np.float64, 1e-12)])
def test_consistent_adjoints_pass(dtype, tolerance):
    assert Matrix().check_adjoint(dtype=dtype) < tolerance
    X, Y = weighted_spaces()
    weighted = Matrix((DOMAIN_WEIGHTS, CODOMAIN_WEIGHTS), domain=X, codomain=Y)
    assert weighted.check_adjoint(dtype=dtype) < tolerance


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_euclidean_adjoint_in_weighted_spaces_fails(dtype):
    X, Y = weighted_spaces()
    assert Matrix(domain=X, codomain=Y).check_adjoint(dtype=dtype) > 1e-1


def test_uniform_rescaling_of_both_spaces_keeps_the_adjoint():
    # A common constant factor cancels from both sides of the adjoint test.
    operator = Matrix(domain=Space((5,), weights=0.3), codomain=Space((12,), weights=0.3))
    assert operator.check_adjoint(dtype=np.float64) < 1e-12


def test_result_is_deterministic_by_default():
    X, Y = weighted_spaces()
    operator = Matrix(domain=X, codomain=Y)
    assert operator.check_adjoint() == operator.check_adjoint()


def test_ambiguous_spaces_are_rejected():
    X, Y = weighted_spaces()
    weighted = Matrix((DOMAIN_WEIGHTS, CODOMAIN_WEIGHTS), domain=X, codomain=Y)
    with pytest.raises(ValueError, match="does not declare its domain and codomain"):
        (weighted + Matrix()).check_adjoint()


class DeviceScale(Operator):
    """Image operator that only accepts device arrays."""

    def apply_to(self, argument, output=None, **kwargs):
        result = clarray.empty_like(argument)
        result[...] = 2 * argument
        return result

    apply_adjoint_to = apply_to


@pytest.fixture(scope="module")
def queue():
    return cl.CommandQueue(cl.create_some_context(interactive=False))


def test_device_operators_are_checked_on_the_device(queue):
    R = Radon(image_domain=16, angles=10)
    S = DeviceScale(
        input_shape=R.input_shape,
        output_shape=R.input_shape,
        domain=R.domain,
        codomain=R.domain,
    )
    assert (R * S).check_adjoint(queue=queue) < 1e-4
    assert R.check_adjoint(queue=queue) < 1e-4
