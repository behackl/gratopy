"""Generic operators and expression nodes for operator algebra."""

from __future__ import annotations

from math import prod
from numbers import Integral, Number
from typing import Any, Literal, Sequence
from weakref import ReferenceType, ref

import numpy as np
import numpy.typing as npt

from gratopy.operator.space import Space
from gratopy.utilities import Numeric


def _compute_sum_shapes(
    operands: Sequence[Operator],
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    input_shape = None
    output_shape = None

    for op in operands:
        if op.input_shape is not None:
            if input_shape is not None and input_shape != op.input_shape:
                raise ValueError(
                    f"Input shape mismatch in sum: expected {input_shape}, "
                    f"but {op} has input shape {op.input_shape}"
                )
            input_shape = op.input_shape

        if op.output_shape is not None:
            if output_shape is not None and output_shape != op.output_shape:
                raise ValueError(
                    f"Output shape mismatch in sum: expected {output_shape}, "
                    f"but {op} has output shape {op.output_shape}"
                )
            output_shape = op.output_shape

    return input_shape, output_shape


def _compute_product_shapes(
    operands: Sequence[Operator],
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    for left, right in zip(operands, operands[1:]):
        if left.input_shape is not None and right.output_shape is not None:
            if left.input_shape != right.output_shape:
                raise ValueError(
                    f"Shape mismatch in composition: {left} expects input "
                    f"{left.input_shape}, but {right} produces {right.output_shape}"
                )

    input_shape = operands[-1].input_shape
    output_shape = operands[0].output_shape
    return input_shape, output_shape


def _common_space(spaces: Sequence[Space | None], description: str) -> Space | None:
    """Return the common specified space, raising if specified spaces differ."""
    common = None
    for space in spaces:
        if space is None:
            continue
        if common is not None and space != common:
            raise ValueError(f"{description} mismatch: {common} and {space}")
        common = space
    return common


def _validate_product_spaces(operands: Sequence[Operator]) -> None:
    for left, right in zip(operands, operands[1:]):
        _common_space(
            [left.domain, right.codomain],
            f"Space mismatch in composition of {left} and {right}",
        )


def _to_host(array: Any) -> np.ndarray:
    """Return a NumPy copy of a NumPy or device (e.g. pyopencl) array."""
    if isinstance(array, np.ndarray):
        return array
    get = getattr(array, "get", None)
    return np.asarray(get() if callable(get) else array)


def _adjoint_mismatch(
    lhs: float,
    rhs: float,
    ax_norm: float,
    y_norm: float,
    x_norm: float,
    aty_norm: float,
) -> float:
    """Return the relative mismatch of the adjoint test.

    ``lhs`` is :math:`\\langle Ax, y\\rangle_Y`, ``rhs`` is
    :math:`\\langle x, A^* y\\rangle_X`; the result is
    :math:`|\\mathrm{lhs} - \\mathrm{rhs}| / \\max(|\\mathrm{lhs}|, |\\mathrm{rhs}|)`.
    The norms of the four vectors are passed for alternative normalizations
    and diagnostics.
    """
    # Normalizing by the larger of the two values makes a wrong adjoint show
    # up as an O(1) mismatch independently of the problem size (normalizing by
    # the Cauchy-Schwarz bound would shrink it like 1/sqrt(size)). The price is
    # a heavier tail when lhs happens to be close to 0 for the random x, y, so
    # the tolerance must not be too tight.
    scale = max(abs(lhs), abs(rhs))
    if scale == 0:
        return 0.0  # both sides agree (e.g. A = 0)
    return abs(lhs - rhs) / scale


def _default_adjoint_rtol(dtype: np.dtype) -> float:
    """Return the tolerance on :func:`_adjoint_mismatch` for ``dtype``."""
    # Wrong weightings give mismatches of about 1. Correct float32 adjoints
    # stay around 1e-5 but have a tail (random lhs close to 0) that grows with
    # the problem size; 1e-2 leaves room for it. Complex dtypes use the
    # precision of their real part.
    if np.finfo(dtype).dtype.itemsize >= 8:
        return 1e-8
    return 1e-2


def _scalar_repr(scalar: Numeric) -> str:
    scalar_repr = repr(scalar)
    if scalar < 0:
        return f"({scalar_repr})"
    return scalar_repr


def _with_return_event(result: Any, return_event: bool) -> Any:
    if not return_event:
        return result
    return result, list(getattr(result, "events", []))


class Operator:
    """Base class for operators and operator expression nodes."""

    #: Whether :attr:`T` is known to be the adjoint in the default spaces, so
    #: that :meth:`ensure_adjoint` can skip the numerical test. Set by
    #: gratopy's own leaf operators; expression nodes derive it from their
    #: operands.
    _adjoint_trusted: bool = False
    # Whether ``domain`` and ``codomain`` may be assigned. Expression nodes and
    # projection operators derive their spaces and disallow assignment.
    _spaces_assignable = True

    def __init__(
        self,
        name: str | None = None,
        state: dict[str, Any] | None = None,
        input_shape: tuple[int, ...] | None = None,
        output_shape: tuple[int, ...] | None = None,
        domain: Space | None = None,
        codomain: Space | None = None,
    ):
        self.name = self.__class__.__name__ if name is None else name
        self.state = {} if state is None else state
        self.input_shape = input_shape
        self.output_shape = output_shape
        self._domain = self._checked_space(domain, input_shape, "domain")
        self._codomain = self._checked_space(codomain, output_shape, "codomain")
        self._adjoint_ref: ReferenceType[AdjointOperator] | None = None
        # (domain, codomain, dtype) combinations in which T passed the test.
        self._adjoint_verified: set[tuple[Any, Any, np.dtype]] = set()

    def __getstate__(self) -> dict[str, Any]:
        """Return serializable state without the weak adjoint memoization."""
        state = self.__dict__.copy()
        state["_adjoint_ref"] = None
        # Leases on compiled OpenCL programs (see gratopy.operator._program_cache).
        state.pop("_kernel_leases", None)
        return state

    @property
    def domain(self) -> Space | None:
        """Space of the operator's inputs, or ``None`` if unspecified.

        The adjoint :attr:`T` refers to the inner products of :attr:`domain`
        and :attr:`codomain`. Solvers treat an operator without any declared
        spaces as Euclidean, and reject expressions that combine operators
        with and without declared spaces.
        """
        if self._domain is not None:
            return self._domain
        return self._default_domain()

    @domain.setter
    def domain(self, space: Space | None) -> None:
        self._ensure_spaces_assignable()
        self._domain = self._checked_space(space, self.input_shape, "domain")

    @property
    def codomain(self) -> Space | None:
        """Space of the operator's outputs, or ``None`` if unspecified."""
        if self._codomain is not None:
            return self._codomain
        return self._default_codomain()

    @codomain.setter
    def codomain(self, space: Space | None) -> None:
        self._ensure_spaces_assignable()
        self._codomain = self._checked_space(space, self.output_shape, "codomain")

    def _ensure_spaces_assignable(self) -> None:
        if not self._spaces_assignable:
            raise AttributeError(
                f"The spaces of {self!r} are derived and cannot be assigned."
            )

    @staticmethod
    def _checked_space(
        space: Space | None,
        shape: tuple[int, ...] | None,
        description: str,
    ) -> Space | None:
        if space is None:
            return None
        if not isinstance(space, Space):
            raise TypeError(f"{description} must be a Space, got {type(space)}")
        if shape is not None and space.shape != tuple(shape):
            raise ValueError(
                f"The {description} shape {space.shape} does not match the "
                f"operator shape {tuple(shape)}"
            )
        return space

    def _default_domain(self) -> Space | None:
        """Return the domain implied by the operator's structure."""
        return None

    def _default_codomain(self) -> Space | None:
        """Return the codomain implied by the operator's structure."""
        return None

    def _is_adjoint_trusted(self) -> bool:
        """Return whether :attr:`T` is the adjoint without a numerical test.

        Explicitly set spaces are never trusted: the setters change the inner
        products but not what :attr:`T` computes.
        """
        return self._adjoint_trusted and self._domain is None and self._codomain is None

    def _adjoint_key(self, dtype: npt.DTypeLike) -> tuple[Any, Any, np.dtype]:
        return (self.domain, self.codomain, np.dtype(dtype))

    def mark_adjoint_verified(self, dtype: npt.DTypeLike = np.float32) -> None:
        """Declare :attr:`T` the adjoint in the current spaces for ``dtype``.

        This skips :meth:`ensure_adjoint` (e.g., for an operator whose adjoint
        is knowingly approximate). Changing :attr:`domain` or :attr:`codomain`
        afterwards revokes it.
        """
        self._adjoint_verified.add(self._adjoint_key(dtype))

    def check_adjoint(
        self,
        dtype: npt.DTypeLike = np.float32,
        input_shape: tuple[int, ...] | None = None,
        queue: Any | None = None,
        rng: np.random.Generator | None = None,
        adjoint: Operator | None = None,
        **kwargs: Any,
    ) -> float:
        """Return the relative mismatch of the randomized adjoint test.

        Compares :math:`\\langle Ax, y\\rangle_Y` with
        :math:`\\langle x, A^* y\\rangle_X` for random ``x`` and ``y``, where
        :math:`A^*` is :attr:`T` and the inner products are those of
        :attr:`domain` and :attr:`codomain` (Euclidean if unspecified). This
        costs one application of the operator and one of its adjoint.

        A mismatch of order 1 means :attr:`T` is not the adjoint in these
        spaces, e.g., because a weighting is missing or a space was changed
        with the :attr:`domain`/:attr:`codomain` setters.
        """
        if input_shape is None:
            input_shape = self.input_shape
        if input_shape is None:
            raise ValueError(
                "Cannot check the adjoint without an input shape. Set "
                "operator.input_shape or pass input_shape explicitly."
            )
        if rng is None:
            # Fixed seed: deterministic, and the caller's rng is not consumed.
            rng = np.random.default_rng(0)
        if adjoint is None:
            adjoint = self.T
        if queue is not None:
            kwargs["queue"] = queue

        dtype = np.dtype(dtype)
        x = np.asarray(rng.standard_normal(input_shape), dtype=dtype)
        ax = _to_host(self.apply_to(x, **kwargs))
        y = np.asarray(rng.standard_normal(ax.shape), dtype=dtype)
        aty = _to_host(adjoint.apply_to(y, **kwargs))

        X = self.domain if self.domain is not None else Space(tuple(input_shape))
        Y = self.codomain if self.codomain is not None else Space(ax.shape)
        # Space.inner is bilinear; conjugating the second argument gives the
        # (real part of the) Hermitian inner product for complex operators.
        lhs = Y.inner(ax, np.conj(y))
        rhs = X.inner(x, np.conj(aty))
        return _adjoint_mismatch(
            lhs, rhs, Y.norm(np.abs(ax)), Y.norm(y), X.norm(x), X.norm(np.abs(aty))
        )

    def ensure_adjoint(
        self,
        dtype: npt.DTypeLike = np.float32,
        input_shape: tuple[int, ...] | None = None,
        queue: Any | None = None,
        rtol: float | None = None,
        adjoint: Operator | None = None,
        **kwargs: Any,
    ) -> None:
        """Raise if :attr:`T` is not the adjoint in :attr:`domain`/:attr:`codomain`.

        Operators built only from gratopy's projection operators,
        ``IDENTITY`` and ``ZERO`` are skipped. Otherwise :meth:`check_adjoint`
        runs once per combination of spaces and dtype; the result is cached,
        and changing a space triggers a new test.
        """
        if self._is_adjoint_trusted():
            return
        key = self._adjoint_key(dtype)
        if key in self._adjoint_verified:
            return

        mismatch = self.check_adjoint(
            dtype=dtype, input_shape=input_shape, queue=queue, adjoint=adjoint, **kwargs
        )
        tolerance = _default_adjoint_rtol(np.dtype(dtype)) if rtol is None else rtol
        if not mismatch <= tolerance:
            raise ValueError(
                f"{self}.T is not the adjoint with respect to domain "
                f"{self.domain} and codomain {self.codomain} (relative mismatch "
                f"{mismatch:.1e}, tolerance {tolerance:.1e}; unspecified spaces "
                "are Euclidean). Declare the spaces T is adjoint in, or skip "
                "the test with check_adjoint=False or mark_adjoint_verified()."
            )
        self._adjoint_verified.add(key)

    def _repr_name_(self) -> str:
        """Return the operator name used in string representations."""
        return self.name

    def __repr__(self) -> str:
        return self._repr_name_()

    def __eq__(self, other: Any) -> bool:
        """Check structural equality of two operators."""
        if not isinstance(other, Operator):
            return False
        return all(
            [
                type(self) is type(other),
                self.name == other.name,
                self.state == other.state,
                self.input_shape == other.input_shape,
                self.output_shape == other.output_shape,
            ]
        )

    @property
    def T(self) -> Operator:
        """Return a memoized adjoint wrapper referencing this operator."""
        if self._adjoint_ref is not None:
            adjoint = self._adjoint_ref()
            if adjoint is not None:
                return adjoint

        adjoint = AdjointOperator(self)
        self._adjoint_ref = ref(adjoint)
        return adjoint

    def norm_estimate(
        self,
        number_iterations: int = 50,
        dtype: npt.DTypeLike = np.float32,
        input_shape: tuple[int, ...] | None = None,
        algorithm: Literal["poweriteration", "naive"] = "poweriteration",
        queue: Any | None = None,
        rng: np.random.Generator | None = None,
        check_adjoint: bool = True,
        adjoint_rtol: float | None = None,
        **kwargs: Any,
    ) -> float:
        """Estimate the operator norm.

        Parameters
        ----------
        number_iterations:
            Positive number of power-iteration updates.
        dtype:
            Data type of the random starting vector.
        input_shape:
            Shape of the starting vector. Defaults to :attr:`input_shape`.
        algorithm:
            ``"poweriteration"`` applies power iteration to ``self.T * self``,
            normalizing in the norm of :attr:`domain` (Euclidean if
            unspecified). The norm is taken with respect to the operator's
            spaces; if :attr:`T` is the adjoint in these spaces, each iterate
            is a lower bound of the norm (up to rounding).
            ``"naive"`` combines power-iteration estimates for expression leaves
            using the triangle inequality for sums and submultiplicativity for
            compositions. Since leaf norms are themselves estimates, the result
            is a heuristic and not a guaranteed upper bound.
        queue:
            Optional backend-specific execution queue forwarded to applications.
        rng:
            Random-number generator used to construct the starting vector.
        check_adjoint:
            Before power iteration, verify via :meth:`ensure_adjoint` that
            :attr:`T` is the adjoint in the operator's spaces (once per
            operator, spaces and dtype; skipped for gratopy's own operators).
            Raises :class:`ValueError` if not.
        adjoint_rtol:
            Tolerance of that check; defaults to ``1e-2`` for single and
            ``1e-8`` for double precision.
        """
        if algorithm not in {"poweriteration", "naive"}:
            raise ValueError(
                "Unknown norm-estimation algorithm. Expected 'poweriteration' or 'naive'."
            )
        if (
            not isinstance(number_iterations, Integral)
            or isinstance(number_iterations, bool)
            or number_iterations < 1
        ):
            raise ValueError("number_iterations must be a positive integer")
        number_iterations = int(number_iterations)

        exact_norm = self._exact_norm()
        if exact_norm is not None:
            return exact_norm

        if algorithm == "naive":
            naive_norm = self._naive_norm_estimate(
                number_iterations=number_iterations,
                dtype=dtype,
                queue=queue,
                rng=rng,
                check_adjoint=check_adjoint,
                adjoint_rtol=adjoint_rtol,
                **kwargs,
            )
            if naive_norm is not None:
                return naive_norm

        return self._power_iteration_norm_estimate(
            number_iterations=number_iterations,
            dtype=dtype,
            input_shape=input_shape,
            queue=queue,
            rng=rng,
            check_adjoint=check_adjoint,
            adjoint_rtol=adjoint_rtol,
            **kwargs,
        )

    def _exact_norm(self) -> float | None:
        """Return an exact norm when one is available for this operator."""
        return None

    def _naive_norm_estimate(self, **kwargs: Any) -> float | None:
        """Return a structural norm estimate, or defer to power iteration."""
        return None

    def _power_iteration_norm_estimate(
        self,
        number_iterations: int,
        dtype: npt.DTypeLike,
        input_shape: tuple[int, ...] | None,
        queue: Any | None,
        rng: np.random.Generator | None,
        check_adjoint: bool = True,
        adjoint_rtol: float | None = None,
        **kwargs: Any,
    ) -> float:
        if input_shape is None:
            input_shape = self.input_shape
        if input_shape is None:
            raise ValueError(
                "Cannot estimate the norm without an input shape. Set "
                "operator.input_shape or pass input_shape explicitly."
            )

        if input_shape == self.input_shape:
            resolve_spaces(self)  # reject expressions with ambiguous adjoints
        adjoint = self.T
        if check_adjoint:
            self.ensure_adjoint(
                dtype=dtype,
                input_shape=tuple(input_shape),
                queue=queue,
                rtol=adjoint_rtol,
                adjoint=adjoint,
                **kwargs,
            )

        if rng is None:
            rng = np.random.default_rng()

        # .T is self-adjoint with respect to the domain's inner product, so
        # normalizing in that norm makes every iterate a lower bound of the norm.
        if input_shape == self.input_shape:
            space = resolve_spaces(self)[0]
        else:
            space = Space(tuple(input_shape))
        x = np.asarray(rng.standard_normal(input_shape), dtype=dtype)
        x_norm = space.norm(x)
        if x_norm == 0:
            return 0.0
        x = x / x_norm

        estimate = 0.0
        for _ in range(number_iterations):
            y = (
                self.apply_to(x, queue=queue, **kwargs)
                if queue is not None
                else self.apply_to(x, **kwargs)
            )
            z = (
                adjoint.apply_to(y, queue=queue, **kwargs)
                if queue is not None
                else adjoint.apply_to(y, **kwargs)
            )
            z_norm = space.norm(z)
            if z_norm == 0:
                return 0.0

            # Since x is normalized, ||(A.T * A)x|| converges to the dominant
            # eigenvalue of A.T * A. Its square root is the operator norm.
            estimate = float(np.sqrt(z_norm))
            # Cast the host scalar to the array dtype: pyopencl would otherwise
            # build a kernel with a double argument, which fails on devices
            # without cl_khr_fp64.
            x = z * z.dtype.type(1.0 / z_norm)

        return estimate

    def apply_to(
        self,
        argument: Any,
        output: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Apply the operator in its forward direction."""
        raise NotImplementedError(
            "apply_to needs to be implemented in specialized subclasses"
        )

    def apply_adjoint_to(
        self,
        argument: Any,
        output: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Apply the adjoint operator."""
        raise NotImplementedError(
            "apply_adjoint_to needs to be implemented in specialized subclasses"
        )

    def is_composite(self) -> bool:
        """Return whether this is a sum or composition expression node."""
        return isinstance(self, (SumOperator, CompositionOperator))

    def __add__(self, other: Operator) -> Operator:
        """Add another operator to this one."""
        if not isinstance(other, Operator):
            raise TypeError(f"Cannot add {type(other)} to {type(self)}")
        if isinstance(other, _ZeroOperator):
            return self
        if isinstance(self, _ZeroOperator):
            return other
        return SumOperator.create(self, other)

    def __neg__(self) -> Operator:
        """Negate this operator."""
        return ScaledOperator.create(-1, self)

    def __sub__(self, other: Operator) -> Operator:
        """Subtract another operator from this one."""
        return self + (-other)

    def __rmul__(self, other: Numeric) -> Operator:  # type: ignore[misc]
        """Multiply this operator by a scalar."""
        if not isinstance(other, Number):
            return NotImplemented
        return ScaledOperator.create(other, self)  # type: ignore[arg-type]

    def __mul__(self, other: Operator | Any) -> Operator | Any:
        """Compose, scale, or apply this operator."""
        if isinstance(other, Operator):
            if isinstance(self, _ZeroOperator) or isinstance(other, _ZeroOperator):
                return ZERO
            if isinstance(self, _IdentityOperator):
                return other
            if isinstance(other, _IdentityOperator):
                return self
            return CompositionOperator.create(self, other)

        if isinstance(other, Number):
            return ScaledOperator.create(other, self)  # type: ignore[arg-type]

        return self.apply_to(other)


class AdjointOperator(Operator):
    """Adjoint view of an underlying operator."""

    _spaces_assignable = False

    def __init__(self, operand: Operator):
        super().__init__(
            name=f"{operand.name}.T",
            input_shape=operand.output_shape,
            output_shape=operand.input_shape,
        )
        self._operand = operand

    @property
    def operand(self) -> Operator:
        return self._operand

    def _default_domain(self) -> Space | None:
        return self.operand.codomain

    def _default_codomain(self) -> Space | None:
        return self.operand.domain

    def _is_adjoint_trusted(self) -> bool:
        return (
            self._domain is None
            and self._codomain is None
            and self.operand._is_adjoint_trusted()
        )

    @property
    def T(self) -> Operator:
        """The adjoint of an adjoint is the underlying operator."""
        return self.operand

    def __repr__(self) -> str:
        if isinstance(self.operand, (SumOperator, CompositionOperator)):
            return f"({self.operand!r}).T"
        return f"{self.operand!r}.T"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AdjointOperator) and self.operand == other.operand

    def _exact_norm(self) -> float | None:
        return self.operand._exact_norm()

    def _naive_norm_estimate(self, **kwargs: Any) -> float | None:
        return self.operand._naive_norm_estimate(**kwargs)

    def apply_to(self, argument: Any, output: Any | None = None, **kwargs: Any) -> Any:
        return self.operand.apply_adjoint_to(argument, output=output, **kwargs)

    def apply_adjoint_to(
        self, argument: Any, output: Any | None = None, **kwargs: Any
    ) -> Any:
        return self.operand.apply_to(argument, output=output, **kwargs)


class ScaledOperator(Operator):
    """An operator multiplied by a scalar."""

    _spaces_assignable = False

    def __init__(self, scalar: Numeric, operand: Operator):
        super().__init__(
            name=operand.name,
            input_shape=operand.input_shape,
            output_shape=operand.output_shape,
        )
        self._scalar = scalar
        self._operand = operand

    @property
    def scalar(self) -> Numeric:
        return self._scalar

    @property
    def operand(self) -> Operator:
        return self._operand

    def _default_domain(self) -> Space | None:
        return self.operand.domain

    def _default_codomain(self) -> Space | None:
        return self.operand.codomain

    def _is_adjoint_trusted(self) -> bool:
        return (
            self._domain is None
            and self._codomain is None
            and self.operand._is_adjoint_trusted()
        )

    @classmethod
    def create(cls, scalar: Numeric, operand: Operator) -> Operator:
        if scalar == 0 or isinstance(operand, _ZeroOperator):
            return ZERO
        if scalar == 1:
            return operand
        if isinstance(operand, ScaledOperator):
            return cls.create(scalar * operand.scalar, operand.operand)
        return cls(scalar, operand)

    def __repr__(self) -> str:
        operand_repr = repr(self.operand)
        if isinstance(self.operand, SumOperator):
            operand_repr = f"({operand_repr})"
        return f"{_scalar_repr(self.scalar)}*{operand_repr}"

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, ScaledOperator)
            and self.scalar == other.scalar
            and self.operand == other.operand
        )

    def _exact_norm(self) -> float | None:
        operand_norm = self.operand._exact_norm()
        if operand_norm is None:
            return None
        return abs(float(self.scalar)) * operand_norm

    def _naive_norm_estimate(self, **kwargs: Any) -> float:
        return abs(float(self.scalar)) * self.operand.norm_estimate(
            algorithm="naive", **kwargs
        )

    def _apply_scaled(
        self,
        argument: Any,
        output: Any | None,
        adjoint: bool,
        **kwargs: Any,
    ) -> Any:
        return_event = bool(kwargs.pop("return_event", False))
        apply = self.operand.apply_adjoint_to if adjoint else self.operand.apply_to
        result = apply(argument, output=output, **kwargs)

        scalar = np.conjugate(self.scalar) if adjoint else self.scalar
        try:
            scalar = result.dtype.type(scalar)
        except AttributeError:
            pass
        result *= scalar
        return _with_return_event(result, return_event)

    def apply_to(self, argument: Any, output: Any | None = None, **kwargs: Any) -> Any:
        return self._apply_scaled(argument, output, False, **kwargs)

    def apply_adjoint_to(
        self, argument: Any, output: Any | None = None, **kwargs: Any
    ) -> Any:
        return self._apply_scaled(argument, output, True, **kwargs)


class SumOperator(Operator):
    """A sum of operators with compatible input and output shapes."""

    _spaces_assignable = False

    def __init__(self, operands: tuple[Operator, ...]):
        input_shape, output_shape = _compute_sum_shapes(operands)
        _common_space([op.domain for op in operands], "Domain in sum")
        _common_space([op.codomain for op in operands], "Codomain in sum")
        super().__init__(input_shape=input_shape, output_shape=output_shape)
        self._operands = operands

    @property
    def operands(self) -> tuple[Operator, ...]:
        return self._operands

    def _default_domain(self) -> Space | None:
        return _common_space([op.domain for op in self.operands], "Domain in sum")

    def _default_codomain(self) -> Space | None:
        return _common_space([op.codomain for op in self.operands], "Codomain in sum")

    def _is_adjoint_trusted(self) -> bool:
        return (
            self._domain is None
            and self._codomain is None
            and all(op._is_adjoint_trusted() for op in self.operands)
        )

    @classmethod
    def create(cls, *operators: Operator) -> Operator:
        operands: list[Operator] = []
        for operator in operators:
            if isinstance(operator, _ZeroOperator):
                continue
            if isinstance(operator, SumOperator):
                operands.extend(operator.operands)
            else:
                operands.append(operator)
        if not operands:
            return ZERO
        if len(operands) == 1:
            return operands[0]
        return cls(tuple(operands))

    def __repr__(self) -> str:
        return " + ".join(repr(op) for op in self.operands)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, SumOperator) and self.operands == other.operands

    def _naive_norm_estimate(self, **kwargs: Any) -> float:
        return sum(
            operand.norm_estimate(algorithm="naive", **kwargs)
            for operand in self.operands
        )

    def _apply_sum(
        self,
        argument: Any,
        output: Any | None,
        adjoint: bool,
        **kwargs: Any,
    ) -> Any:
        return_event = bool(kwargs.pop("return_event", False))

        def apply(operand: Operator, child_output: Any | None = None) -> Any:
            if adjoint:
                return operand.apply_adjoint_to(
                    argument,
                    output=child_output,
                    **kwargs,
                )
            return operand.apply_to(argument, output=child_output, **kwargs)

        result = apply(self.operands[0], output)
        for operand in self.operands[1:]:
            result += apply(operand)
        return _with_return_event(result, return_event)

    def apply_to(self, argument: Any, output: Any | None = None, **kwargs: Any) -> Any:
        return self._apply_sum(argument, output, False, **kwargs)

    def apply_adjoint_to(
        self, argument: Any, output: Any | None = None, **kwargs: Any
    ) -> Any:
        return self._apply_sum(argument, output, True, **kwargs)


class CompositionOperator(Operator):
    """A composition ``A*B`` representing ``A(B(x))``."""

    _spaces_assignable = False

    def __init__(self, operands: tuple[Operator, ...]):
        input_shape, output_shape = _compute_product_shapes(operands)
        _validate_product_spaces(operands)
        super().__init__(input_shape=input_shape, output_shape=output_shape)
        self._operands = operands

    @property
    def operands(self) -> tuple[Operator, ...]:
        return self._operands

    def _default_domain(self) -> Space | None:
        return self.operands[-1].domain

    def _default_codomain(self) -> Space | None:
        return self.operands[0].codomain

    def _is_adjoint_trusted(self) -> bool:
        return (
            self._domain is None
            and self._codomain is None
            and all(op._is_adjoint_trusted() for op in self.operands)
        )

    @classmethod
    def create(cls, *operators: Operator) -> Operator:
        operands: list[Operator] = []
        for operator in operators:
            if isinstance(operator, _ZeroOperator):
                return ZERO
            if isinstance(operator, _IdentityOperator):
                continue
            if isinstance(operator, CompositionOperator):
                operands.extend(operator.operands)
            else:
                operands.append(operator)
        if not operands:
            return IDENTITY
        if len(operands) == 1:
            return operands[0]
        return cls(tuple(operands))

    def __repr__(self) -> str:
        representations = []
        for operand in self.operands:
            operand_repr = repr(operand)
            if isinstance(operand, SumOperator):
                operand_repr = f"({operand_repr})"
            representations.append(operand_repr)
        return "*".join(representations)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CompositionOperator) and self.operands == other.operands

    def _naive_norm_estimate(self, **kwargs: Any) -> float:
        return prod(
            operand.norm_estimate(algorithm="naive", **kwargs)
            for operand in self.operands
        )

    def _apply_composition(
        self,
        argument: Any,
        output: Any | None,
        adjoint: bool,
        **kwargs: Any,
    ) -> Any:
        return_event = bool(kwargs.pop("return_event", False))
        result = argument
        application_order = self.operands if adjoint else tuple(reversed(self.operands))
        for index, operand in enumerate(application_order):
            child_output = output if index == len(application_order) - 1 else None
            apply = operand.apply_adjoint_to if adjoint else operand.apply_to
            result = apply(result, output=child_output, **kwargs)
        return _with_return_event(result, return_event)

    def apply_to(self, argument: Any, output: Any | None = None, **kwargs: Any) -> Any:
        return self._apply_composition(argument, output, False, **kwargs)

    def apply_adjoint_to(
        self, argument: Any, output: Any | None = None, **kwargs: Any
    ) -> Any:
        return self._apply_composition(argument, output, True, **kwargs)


class _IdentityOperator(Operator):
    """Shape-agnostic identity operator."""

    # Self-adjoint with respect to every inner product.
    _adjoint_trusted = True
    _spaces_assignable = False

    @property
    def T(self) -> Operator:
        return self

    def _exact_norm(self) -> float:
        return 1.0

    def apply_to(self, argument: Any, output: Any | None = None, **kwargs: Any) -> Any:
        result = 1 * argument
        if output is not None:
            output[...] = result
            return output
        return result

    def apply_adjoint_to(
        self, argument: Any, output: Any | None = None, **kwargs: Any
    ) -> Any:
        return self.apply_to(argument, output=output, **kwargs)


class _ZeroOperator(Operator):
    """Shape-agnostic zero operator."""

    # Self-adjoint with respect to every inner product.
    _adjoint_trusted = True
    _spaces_assignable = False

    @property
    def T(self) -> Operator:
        return self

    def _exact_norm(self) -> float:
        return 0.0

    def apply_to(self, argument: Any, output: Any | None = None, **kwargs: Any) -> Any:
        if output is not None:
            output[...] = 0
            return output

        try:
            return 0 * argument  # type: ignore
        except TypeError:
            pass

        try:
            return np.zeros_like(argument)
        except (ValueError, TypeError):
            pass

        raise TypeError(f"Cannot apply zero operator to {type(argument)}")

    def apply_adjoint_to(
        self, argument: Any, output: Any | None = None, **kwargs: Any
    ) -> Any:
        return self.apply_to(argument, output=output, **kwargs)


IDENTITY = _IdentityOperator(name="[Id]")
ZERO = _ZeroOperator(name="[0]")


def _leaf_operators(operator: Operator) -> list[Operator]:
    """Return the leaf operators of an expression, without IDENTITY and ZERO."""
    if isinstance(operator, (AdjointOperator, ScaledOperator)):
        return _leaf_operators(operator.operand)
    if isinstance(operator, (SumOperator, CompositionOperator)):
        return [leaf for op in operator.operands for leaf in _leaf_operators(op)]
    if isinstance(operator, (_IdentityOperator, _ZeroOperator)):
        return []
    return [operator]


def resolve_spaces(operator: Operator) -> tuple[Space, Space]:
    """Return the domain and codomain in which ``operator.T`` is the adjoint.

    Operators without any declared spaces are Euclidean. Expressions that
    combine operators with and without declared spaces are rejected, as the
    inner products their adjoint refers to are then ambiguous.

    :raises ValueError: If the declarations of the leaf operators are mixed,
        or if the operator has no input or output shape.
    """
    leaves = _leaf_operators(operator)
    # Partition by identity: Operator.__eq__ is structural, so distinct leaves
    # (e.g. two instances of one custom class) may compare equal.
    is_declared = [
        leaf.domain is not None and leaf.codomain is not None for leaf in leaves
    ]
    if not all(is_declared) and any(is_declared):
        undeclared = [leaf for leaf, ok in zip(leaves, is_declared) if not ok]
        names = ", ".join(sorted({repr(leaf) for leaf in undeclared}))
        declared = next(leaf for leaf, ok in zip(leaves, is_declared) if ok)
        raise ValueError(
            f"Cannot determine the inner products of {operator!r}: {names} "
            f"does not declare its domain and codomain, but {declared!r} does. "
            "Declare the spaces of custom operators that are combined with "
            "projection operators, e.g. "
            "Operator(..., domain=R.domain, codomain=R.domain)."
        )

    domain, codomain = operator.domain, operator.codomain
    if domain is not None and codomain is not None:
        return domain, codomain
    if operator.input_shape is None or operator.output_shape is None:
        raise ValueError(
            f"Cannot determine the spaces of {operator!r}: its input and "
            "output shapes are unknown."
        )
    return (
        domain or Space(tuple(operator.input_shape)),
        codomain or Space(tuple(operator.output_shape)),
    )
