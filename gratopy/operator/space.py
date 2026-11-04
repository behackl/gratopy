"""Function spaces of operators.

A :class:`Space` describes the Hilbert space an operator acts on or maps into:
a discretized :math:`L^2` space whose inner product carries positive weights.
Projection operators use weighted spaces (pixel area for images, detector
pixel width times angle weights for sinograms); their adjoints :attr:`T` are
adjoint with respect to exactly these inner products.

Arrays may carry trailing dimensions beyond :attr:`Space.shape` (e.g., a stack
of slices); these are summed over without additional weights, consistent with
the slicewise application of operators.
"""

from __future__ import annotations

from math import prod
from typing import Any, Literal, NoReturn

import numpy as np
import numpy.typing as npt
import pyopencl as cl
import pyopencl.array as clarray
from pyopencl.reduction import ReductionKernel

from gratopy.operator._program_cache import (
    _ProgramBundle,
    _ProgramLeaseKey,
    acquire_leased,
)


class Space:
    """Discretized :math:`L^2` space with a weighted inner product.

    The inner product is :math:`\\langle x, y \\rangle = \\sum_i w_i x_i y_i`
    with positive weights :math:`w`. Spaces are immutable.

    **Parameters**

    ``shape``:
        Base shape of the elements of the space.
    ``weights``:
        Positive weights, as a scalar or an array broadcastable to ``shape``
        (NumPy broadcasting rules). Defaults to 1, the Euclidean space.

    **Examples**

    Images with pixel area 0.25, and sinograms with detector pixel width 0.1
    and one weight per angle (the last axis):

    >>> import numpy as np
    >>> from gratopy.operator import Space
    >>> images = Space((64, 64), weights=0.25)
    >>> sinograms = Space((91, 3), weights=0.1 * np.array([1.0, 2.0, 1.0]))
    >>> sinograms.inner(np.ones((91, 3)), np.ones((91, 3)))
    36.4
    """

    __slots__ = (
        "shape",
        "_weights",
        "_uniform_weight",
        "_device_weights",
        "_kernel_leases",
    )

    shape: tuple[int, ...]
    _weights: np.ndarray
    _uniform_weight: float | None
    _device_weights: dict[tuple[cl.Context, np.dtype, str], clarray.Array]
    _kernel_leases: dict[_ProgramLeaseKey, _ProgramBundle]

    def __init__(self, shape: tuple[int, ...], weights: npt.ArrayLike = 1.0) -> None:
        shape = tuple(int(n) for n in shape)
        canonical = _canonical_weights(weights, shape)
        uniform = float(canonical.flat[0]) if canonical.size == 1 else None
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "_weights", canonical)
        object.__setattr__(self, "_uniform_weight", uniform)
        object.__setattr__(self, "_device_weights", {})
        object.__setattr__(self, "_kernel_leases", {})

    @property
    def weights(self) -> np.ndarray:
        """Read-only weights, broadcastable to :attr:`shape`.

        The array has one axis per axis of :attr:`shape`; axes along which
        the weights are constant have length 1.
        """
        return self._weights

    @property
    def uniform_weight(self) -> float | None:
        """The common weight if all weights are equal, otherwise ``None``."""
        return self._uniform_weight

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise AttributeError("Space objects are immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("Space objects are immutable")

    def __getstate__(self) -> dict[str, Any]:
        """Return pickle/deepcopy state without device buffers and leases."""
        return {"shape": self.shape, "weights": self._weights}

    def __setstate__(self, state: dict[str, Any]) -> None:
        Space.__init__(self, state["shape"], state["weights"])

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Space):
            return NotImplemented
        return self.shape == other.shape and bool(
            np.array_equal(self._weights, other._weights)
        )

    def __hash__(self) -> int:
        return hash((self.shape, self._weights.shape, self._weights.tobytes()))

    def __repr__(self) -> str:
        if self._uniform_weight is not None:
            weights = repr(self._uniform_weight)
        else:
            weights = f"<array of shape {self._weights.shape}>"
        return f"Space(shape={self.shape}, weights={weights})"

    def inner(self, x: Any, y: Any) -> float:
        """Return the inner product of ``x`` and ``y``.

        ``x`` and ``y`` must both be NumPy arrays or both be
        :class:`pyopencl.array.Array` instances of the same shape (and, on the
        device, the same memory order).
        """
        self._validate(x, y)
        if isinstance(x, clarray.Array):
            return self._device_inner(x, y)

        product = np.asarray(x) * np.asarray(y)
        if self._uniform_weight is not None:
            return self._uniform_weight * float(np.real(product.sum()))
        extra_dims = product.ndim - len(self.shape)
        weights = self._weights.reshape(self._weights.shape + (1,) * extra_dims)
        return float(np.real((product * weights).sum()))

    def norm(self, x: Any) -> float:
        """Return the norm of ``x`` induced by :meth:`inner`."""
        return float(np.sqrt(max(self.inner(x, x), 0.0)))

    def _validate(self, x: Any, y: Any) -> None:
        if isinstance(x, clarray.Array) != isinstance(y, clarray.Array):
            raise TypeError(
                "Both arguments must be NumPy arrays or both pyopencl arrays."
            )
        if x.shape != y.shape:
            raise ValueError(f"Shape mismatch: {x.shape} and {y.shape}")
        if tuple(x.shape[: len(self.shape)]) != self.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.shape} (plus trailing "
                f"dimensions), got {x.shape}"
            )

    def _device_inner(self, x: clarray.Array, y: clarray.Array) -> float:
        if x.dtype != y.dtype:
            raise ValueError(f"dtype mismatch: {x.dtype} and {y.dtype}")
        order: Literal["C", "F"]
        if x.flags.c_contiguous and y.flags.c_contiguous:
            order = "C"
        elif x.flags.f_contiguous and y.flags.f_contiguous:
            order = "F"
        else:
            raise ValueError("Both arrays must be contiguous with the same order.")

        if self._uniform_weight is not None:
            return self._uniform_weight * float(np.real(clarray.vdot(x, y).get()))

        queue = x.queue
        assert queue is not None
        weights, stride, count = self._device_weight_layout(queue, x, order)
        kernel = self._weighted_dot_kernel(queue.context, x.dtype)
        result = kernel(
            x,
            y,
            weights,
            np.int64(stride),
            np.int64(count),
            queue=queue,
            return_event=False,
        )
        return float(result.get())

    def _device_weight_layout(
        self, queue: cl.CommandQueue, x: clarray.Array, order: Literal["C", "F"]
    ) -> tuple[clarray.Array, int, int]:
        """Return device weights ``w``, ``stride`` and ``count`` such that the
        weight of flat element ``i`` of ``x`` is ``w[(i / stride) % count]``.
        """
        varying = [axis for axis, n in enumerate(self._weights.shape) if n > 1]
        if len(varying) == 1:
            # Weights along a single axis: upload only that vector.
            (axis,) = varying
            host = self._weights.ravel()
            faster_axes = x.shape[:axis] if order == "F" else x.shape[axis + 1 :]
            stride, count, layout = prod(faster_axes), self.shape[axis], "axis"
        else:
            host = np.broadcast_to(self._weights, self.shape).ravel(order=order)
            extra = x.shape[len(self.shape) :]
            stride = 1 if order == "F" else prod(extra)
            count, layout = prod(self.shape), order

        key = (queue.context, x.dtype, layout)
        weights = self._device_weights.get(key)
        if weights is None:
            weights = clarray.to_device(queue, host.astype(x.dtype))
            self._device_weights[key] = weights
        return weights, stride, count

    def _weighted_dot_kernel(
        self, context: cl.Context, dtype: np.dtype
    ) -> ReductionKernel:
        """Return a reduction computing ``sum_i x[i] y[i] w[(i / stride) % count]``.

        The kernel is shared through the program cache and leased by this
        space, so it is released together with the space.
        """
        dtype = np.dtype(dtype)
        ctype = {np.dtype(np.float32): "float", np.dtype(np.float64): "double"}[dtype]

        def build() -> ReductionKernel:
            return ReductionKernel(
                context,
                dtype_out=dtype,
                neutral="0",
                reduce_expr="a+b",
                map_expr="x[i] * y[i] * w[(i / weight_stride) % weight_count]",
                arguments=(
                    f"const {ctype} *x, const {ctype} *y, const {ctype} *w, "
                    "const long weight_stride, const long weight_count"
                ),
                name="gratopy_weighted_dot",
            )

        bundle = acquire_leased(
            self._kernel_leases,
            ("weighted_dot", context, dtype),
            (context, f"gratopy_weighted_dot:{dtype.name}", None),
            build,
        )
        return bundle.program


def _canonical_weights(weights: npt.ArrayLike, shape: tuple[int, ...]) -> np.ndarray:
    """Validate weights and return them with one axis per axis of ``shape``,
    collapsing axes along which they are constant to length 1.
    """
    array = np.array(weights, dtype=np.float64)
    try:
        broadcast_shape = np.broadcast_shapes(array.shape, shape)
    except ValueError:
        broadcast_shape = None
    if broadcast_shape != shape:
        raise ValueError(
            f"weights of shape {array.shape} are not broadcastable to {shape}"
        )
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("weights must be positive and finite")

    array = array.reshape((1,) * (len(shape) - array.ndim) + array.shape)
    for axis in range(array.ndim):
        if array.shape[axis] > 1 and np.all(array == array.take([0], axis=axis)):
            array = array.take([0], axis=axis)
    array = np.ascontiguousarray(array)
    array.flags.writeable = False
    return array
