"""Iterative least-squares reconstruction: Landweber and conjugate gradients."""

from __future__ import annotations

from typing import Any

import pyopencl as cl

from gratopy.operator.base import Operator, resolve_spaces
from gratopy.reconstruction._common import (
    Callback,
    check_iterations,
    check_positive,
    initial_guess,
    notify,
    prepare_data,
    resolve_operator_norm,
    verify_adjoint,
)
from gratopy.reconstruction._vector import axpby, zeros


def landweber(
    operator: Operator,
    data: Any,
    number_iterations: int = 100,
    *,
    w: float = 1.0,
    x0: Any | None = None,
    operator_norm: float | None = None,
    queue: cl.CommandQueue | None = None,
    callback: Callback | None = None,
    check_adjoint: bool = True,
    adjoint_rtol: float | None = None,
) -> Any:
    """Landweber iteration (also known as SIRT) [L1951]_.

    Approximates a solution of :math:`\\min_x \\frac12\\|Ax - f\\|^2` via the
    iteration :math:`x \\leftarrow x - \\frac{w}{\\|A\\|^2} A^*(Ax - f)`, where
    :math:`A^*` is ``operator.T``.

    :param operator: The forward operator :math:`A`.
    :param data: The data :math:`f` in the operator's output space, possibly
        with trailing slice dimensions.
    :param number_iterations: Number of iterations.
    :param w: Relaxation parameter relative to :math:`\\|A\\|^{-2}`; the
        iteration converges for :math:`0 < w < 2`.
    :param x0: Initial guess (zeros if :obj:`None`); it is not modified.
    :param operator_norm: The norm :math:`\\|A\\|`. If :obj:`None`, it is
        estimated once via :meth:`~gratopy.operator.Operator.norm_estimate`;
        passing it avoids repeated estimates across calls.
    :param queue: OpenCL queue; inferred from ``data`` or ``x0`` if omitted.
        Required for NumPy data with OpenCL operators.
    :param callback: Called as ``callback(iteration, x)`` after every
        iteration; returning :obj:`True` stops the iteration. ``x`` is the
        current iterate (owned by the solver; copy it to keep it).
    :param check_adjoint: Verify once that ``operator.T`` is the adjoint in
        the operator's spaces and raise :class:`ValueError` otherwise (see
        :meth:`~gratopy.operator.Operator.ensure_adjoint`). Skipped for
        gratopy's own operators.
    :param adjoint_rtol: Tolerance of that check; defaults to ``1e-2`` for
        float32 and ``1e-8`` for float64 data.
    :return: The reconstruction, a device array if a queue is available.

    """
    number_iterations = check_iterations(number_iterations)
    w = check_positive(w, "w")
    data = prepare_data(operator, data, queue=queue, x0=x0)
    resolve_spaces(operator)  # reject expressions with ambiguous adjoints
    verify_adjoint(operator, data, check_adjoint, adjoint_rtol)
    x = initial_guess(operator, data, x0)
    step = w / resolve_operator_norm(operator, operator_norm, data) ** 2

    adjoint = operator.T
    residual = zeros(data.shape, like=data)
    gradient = zeros(x.shape, like=x)
    for iteration in range(1, number_iterations + 1):
        residual = operator.apply_to(x, output=residual)
        axpby(1.0, residual, -1.0, data, owner=operator)  # residual = Ax - f
        gradient = adjoint.apply_to(residual, output=gradient)
        axpby(1.0, x, -step, gradient, owner=operator)
        if notify(callback, iteration, x):
            break
    return x


def conjugate_gradients(
    operator: Operator,
    data: Any,
    number_iterations: int = 20,
    *,
    epsilon: float = 0.0,
    x0: Any | None = None,
    queue: cl.CommandQueue | None = None,
    callback: Callback | None = None,
    check_adjoint: bool = True,
    adjoint_rtol: float | None = None,
) -> Any:
    """Conjugate gradients for the normal equations (CGLS) [HS1952]_.

    Approximates a solution of :math:`\\min_x \\|Ax - f\\|^2`. Inner products
    are taken in ``operator.domain`` and ``operator.codomain``, i.e., in the
    same weighted spaces in which ``operator.T`` is the adjoint. Operators
    without declared spaces are Euclidean; expressions mixing operators with
    and without declared spaces raise a :class:`ValueError`.

    Starting from ``x0``, the iteration converges to the least-squares
    solution closest to ``x0`` (the minimum-norm solution for ``x0 = 0``).
    Restarting with a previous result as ``x0`` is not equivalent to
    continuing the iteration, as the search directions are lost.

    :param operator: The forward operator :math:`A`.
    :param data: The data :math:`f` in the operator's output space, possibly
        with trailing slice dimensions.
    :param number_iterations: Maximal number of iterations.
    :param epsilon: The iteration stops once the relative residual
        :math:`\\|Ax - f\\| / \\|f\\|` (in the codomain norm) is below
        ``epsilon``.
    :param x0: Initial guess (zeros if :obj:`None`); it is not modified.
    :param queue: OpenCL queue; inferred from ``data`` or ``x0`` if omitted.
    :param callback: Called as ``callback(iteration, x)`` after every
        iteration; returning :obj:`True` stops the iteration.
    :param check_adjoint: Verify once that ``operator.T`` is the adjoint in
        the operator's spaces and raise :class:`ValueError` otherwise (see
        :meth:`~gratopy.operator.Operator.ensure_adjoint`). Skipped for
        gratopy's own operators.
    :param adjoint_rtol: Tolerance of that check; defaults to ``1e-2`` for
        float32 and ``1e-8`` for float64 data.
    :return: The reconstruction, a device array if a queue is available.

    """
    number_iterations = check_iterations(number_iterations)
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    data = prepare_data(operator, data, queue=queue, x0=x0)
    X, Y = resolve_spaces(operator)
    verify_adjoint(operator, data, check_adjoint, adjoint_rtol)
    x = initial_guess(operator, data, x0)

    adjoint = operator.T
    residual = operator.apply_to(x, output=zeros(data.shape, like=data))
    axpby(-1.0, residual, 1.0, data, owner=operator)  # residual = f - Ax
    gradient = adjoint.apply_to(residual, output=zeros(x.shape, like=x))
    direction = gradient.copy()
    image = zeros(data.shape, like=data)

    data_norm = Y.norm(data)
    gradient_norm_sq = X.inner(gradient, gradient)
    for iteration in range(1, number_iterations + 1):
        image = operator.apply_to(direction, output=image)
        image_norm_sq = Y.inner(image, image)
        if gradient_norm_sq == 0 or image_norm_sq == 0:
            break  # x already solves the normal equations

        alpha = gradient_norm_sq / image_norm_sq
        axpby(1.0, x, alpha, direction, owner=operator)
        axpby(1.0, residual, -alpha, image, owner=operator)
        gradient = adjoint.apply_to(residual, output=gradient)
        new_gradient_norm_sq = X.inner(gradient, gradient)
        beta = new_gradient_norm_sq / gradient_norm_sq
        gradient_norm_sq = new_gradient_norm_sq
        axpby(beta, direction, 1.0, gradient, owner=operator)

        if notify(callback, iteration, x):
            break
        relative_residual = Y.norm(residual) / (data_norm if data_norm > 0 else 1.0)
        if relative_residual < epsilon:
            break
    return x
