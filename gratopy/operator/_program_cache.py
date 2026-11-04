"""Shared cache of compiled OpenCL programs with owner-held leases.

Compiled programs (and pyopencl's generated elementwise/reduction kernels)
are interned in a weak registry, :data:`_PROGRAM_CACHE`, so that equal
compilation inputs are compiled once per context. The registry does not keep
them alive: each user ("owner", e.g. an operator or a :class:`Space`) holds
strong leases on the bundles it uses. A bundle, and with it its OpenCL
context, is released once the last owner using it is gone.

This module only depends on pyopencl so that both operators and spaces can use
it without circular imports.
"""

from __future__ import annotations

from threading import RLock, local
from typing import Any, Callable, Hashable
from weakref import WeakValueDictionary

import pyopencl as cl

#: (context, content signature, variant) identifying compilation inputs.
_ProgramCacheKey = tuple[cl.Context, str, Hashable]
_ProgramLeaseKey = Hashable

#: Attribute under which owners store their leases.
_LEASES_ATTRIBUTE = "_kernel_leases"


class _ProgramBundle:
    """Compiled program shared by owners through strong leases.

    ``program`` is a :class:`pyopencl.Program` or a pyopencl-generated kernel
    object (e.g. :class:`pyopencl.elementwise.ElementwiseKernel`).
    """

    def __init__(
        self,
        *,
        cache_key: _ProgramCacheKey,
        generation: int,
        program: Any,
    ) -> None:
        self.cache_key = cache_key
        self.generation = generation
        self.program = program
        self._thread_state = local()

    def kernel(self, name: str) -> cl.Kernel:
        """Return a kernel of a :class:`pyopencl.Program` local to the thread."""
        kernels = getattr(self._thread_state, "kernels", None)
        if kernels is None:
            kernels = {}
            self._thread_state.kernels = kernels

        kernel = kernels.get(name)
        if kernel is None:
            kernel = cl.Kernel(self.program, name)
            kernels[name] = kernel
        return kernel


class _ProgramCache:
    """Weak registry interning compiled OpenCL program bundles."""

    def __init__(self) -> None:
        self._bundles: WeakValueDictionary[_ProgramCacheKey, _ProgramBundle] = (
            WeakValueDictionary()
        )
        self._generation = 0
        self._lock = RLock()

    @property
    def generation(self) -> int:
        """Return the current invalidation generation."""
        with self._lock:
            return self._generation

    def acquire(
        self,
        cache_key: _ProgramCacheKey,
        build: Callable[[], Any],
    ) -> _ProgramBundle:
        """Return a shared bundle, compiling it once on a cache miss."""
        with self._lock:
            bundle = self._bundles.get(cache_key)
            if bundle is not None:
                return bundle

            bundle = _ProgramBundle(
                cache_key=cache_key,
                generation=self._generation,
                program=build(),
            )
            self._bundles[cache_key] = bundle
            return bundle

    def invalidate(self) -> None:
        """Invalidate all entries while allowing active leases to finish."""
        with self._lock:
            self._generation += 1
            self._bundles.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._bundles)


_PROGRAM_CACHE = _ProgramCache()


def owner_leases(owner: Any) -> dict[_ProgramLeaseKey, _ProgramBundle]:
    """Return the lease table of ``owner``, creating it on first use.

    Owners must drop this attribute in ``__getstate__``, as bundles hold live
    OpenCL objects.
    """
    leases = owner.__dict__.get(_LEASES_ATTRIBUTE)
    if leases is None:
        leases = {}
        setattr(owner, _LEASES_ATTRIBUTE, leases)
    return leases


def acquire_leased(
    leases: dict[_ProgramLeaseKey, _ProgramBundle],
    lease_key: _ProgramLeaseKey,
    cache_key: _ProgramCacheKey,
    build: Callable[[], Any],
) -> _ProgramBundle:
    """Return the bundle for ``cache_key`` and lease it in ``leases``.

    The owner's lease is reused while it matches ``cache_key`` (e.g., the
    kernel source is unchanged) and the cache has not been invalidated since.
    """
    bundle = leases.get(lease_key)
    if (
        bundle is not None
        and bundle.cache_key == cache_key
        and bundle.generation == _PROGRAM_CACHE.generation
    ):
        return bundle

    bundle = _PROGRAM_CACHE.acquire(cache_key, build)
    leases[lease_key] = bundle
    return bundle
