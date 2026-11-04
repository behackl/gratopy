"""Tests for the randomized adjoint check of operators."""

import numpy as np
import pytest

from gratopy.operator import Operator, Space
from gratopy.operator.base import IDENTITY, ZERO
from gratopy.reconstruction import conjugate_gradients

RNG = np.random.default_rng(3)
MATRIX = RNG.standard_normal((12, 5))
DOMAIN_WEIGHTS = RNG.uniform(0.5, 2.0, 5)
CODOMAIN_WEIGHTS = RNG.uniform(0.5, 2.0, 12)


class Matrix(Operator):
    """Matrix operator whose adjoint is weighted by the given (fixed) weights."""

    def __init__(self, adjoint_weights=None, **kwargs):
        super().__init__(input_shape=(5,), output_shape=(12,), **kwargs)
        self.adjoint_weights = adjoint_weights
        self.applications = 0

    def apply_to(self, argument, output=None, **kwargs):
        self.applications += 1
        return MATRIX @ argument

    def apply_adjoint_to(self, argument, output=None, **kwargs):
        self.applications += 1
        if self.adjoint_weights is None:
            return MATRIX.T @ argument
        w_x, w_y = self.adjoint_weights
        return (MATRIX.T @ (w_y * argument)) / w_x


def weighted_spaces():
    return (
        Space((5,), axis_weights=(0, DOMAIN_WEIGHTS)),
        Space((12,), axis_weights=(0, CODOMAIN_WEIGHTS)),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_consistent_adjoints_pass(dtype):
    Matrix().ensure_adjoint(dtype=dtype)
    X, Y = weighted_spaces()
    Matrix((DOMAIN_WEIGHTS, CODOMAIN_WEIGHTS), domain=X, codomain=Y).ensure_adjoint(
        dtype=dtype
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_euclidean_adjoint_in_weighted_spaces_fails(dtype):
    X, Y = weighted_spaces()
    operator = Matrix(domain=X, codomain=Y)
    assert operator.check_adjoint(dtype=dtype) > 1e-2
    with pytest.raises(ValueError, match="not the adjoint"):
        operator.ensure_adjoint(dtype=dtype)


def test_uniform_rescaling_of_both_spaces_keeps_the_adjoint():
    # A common constant factor cancels from both sides of the adjoint test.
    operator = Matrix(domain=Space((5,), scale=0.3), codomain=Space((12,), scale=0.3))
    operator.ensure_adjoint(dtype=np.float64)


def test_result_is_cached_per_spaces_and_dtype():
    operator = Matrix()
    operator.ensure_adjoint(dtype=np.float64)
    count = operator.applications
    operator.ensure_adjoint(dtype=np.float64)
    assert operator.applications == count
    operator.ensure_adjoint(dtype=np.float32)
    assert operator.applications == count + 2


def test_changing_a_space_revokes_the_verification():
    operator = Matrix()
    operator.ensure_adjoint(dtype=np.float64)
    operator.domain, operator.codomain = weighted_spaces()
    with pytest.raises(ValueError, match="not the adjoint"):
        operator.ensure_adjoint(dtype=np.float64)


def test_mark_adjoint_verified_skips_the_check():
    X, Y = weighted_spaces()
    operator = Matrix(domain=X, codomain=Y)
    operator.mark_adjoint_verified(np.float64)
    operator.ensure_adjoint(dtype=np.float64)
    assert operator.applications == 0


def test_expressions_of_trusted_operators_are_not_tested():
    expression = 2 * IDENTITY + ZERO
    assert expression._is_adjoint_trusted()
    assert not (IDENTITY + Matrix().T * Matrix())._is_adjoint_trusted()


def test_expression_with_inherited_space_is_tested_in_that_space():
    # E (Euclidean adjoint, no spaces) inherits the weighted spaces of the sum.
    X, Y = weighted_spaces()
    weighted = Matrix((DOMAIN_WEIGHTS, CODOMAIN_WEIGHTS), domain=X, codomain=Y)
    with pytest.raises(ValueError, match="not the adjoint"):
        (weighted + Matrix()).ensure_adjoint(dtype=np.float64)


def test_explicitly_set_spaces_are_never_trusted():
    from gratopy.operator import Radon

    R = Radon(image_domain=8, angles=6)
    assert R._is_adjoint_trusted()
    R.domain = Space(R.input_shape)
    assert not R._is_adjoint_trusted()


def test_solvers_and_norm_estimate_check_and_allow_opting_out():
    X, Y = weighted_spaces()
    operator = Matrix(domain=X, codomain=Y)
    data = MATRIX @ np.ones(5)
    with pytest.raises(ValueError, match="not the adjoint"):
        conjugate_gradients(operator, data, 3)
    with pytest.raises(ValueError, match="not the adjoint"):
        operator.norm_estimate(dtype=np.float64)
    conjugate_gradients(operator, data, 3, check_adjoint=False)
    operator.norm_estimate(dtype=np.float64, check_adjoint=False)


def test_default_tolerances_and_custom_threshold():
    from gratopy.operator.base import _default_adjoint_rtol

    assert _default_adjoint_rtol(np.dtype(np.float32)) == 1e-2
    assert _default_adjoint_rtol(np.dtype(np.float64)) == 1e-8
    assert _default_adjoint_rtol(np.dtype(np.complex128)) == 1e-8

    X, Y = weighted_spaces()
    operator = Matrix(domain=X, codomain=Y)  # wrong adjoint, mismatch < 1
    operator.ensure_adjoint(dtype=np.float64, rtol=10.0)
    data = MATRIX @ np.ones(5)
    conjugate_gradients(Matrix(domain=X, codomain=Y), data, 3, adjoint_rtol=10.0)
    Matrix(domain=X, codomain=Y).norm_estimate(dtype=np.float64, adjoint_rtol=10.0)
