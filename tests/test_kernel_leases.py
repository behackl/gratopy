"""Lifetime of the compiled programs used by spaces and reconstruction solvers.

Programs are shared through the weak program cache and kept alive by their
owners (the solver's operator, or a Space), so that they and their OpenCL
contexts are released together with the owner.
"""

import copy
import gc
import pickle

import numpy as np
import pyopencl as cl
import pyopencl.array as clarray
import pytest

from gratopy.operator import Radon, Space, invalidate_kernel_cache
from gratopy.operator._program_cache import _PROGRAM_CACHE, owner_leases
from gratopy.reconstruction import conjugate_gradients, total_variation


@pytest.fixture(scope="module")
def device():
    return cl.create_some_context(interactive=False).devices[0]


def cached_contexts():
    """Return the native handles of all contexts with cached programs."""
    return {key[0].int_ptr for key in list(_PROGRAM_CACHE._bundles.keys())}


def run_solvers(device):
    """Create a context and operator, run the solvers, return the context handle."""
    queue = cl.CommandQueue(cl.Context([device]))
    R = Radon(image_domain=8, angles=6)
    data = clarray.to_device(
        queue, np.random.default_rng(0).random(R.output_shape, dtype=np.float32)
    )
    total_variation(R, data, mu=1.0, number_iterations=2, operator_norm=1.0)
    conjugate_gradients(R, data, 2)
    leased = {key[0] for key in owner_leases(R)}
    assert {"total_variation", "axpby"} <= leased
    assert queue.context.int_ptr in cached_contexts()  # cached while R lives
    return queue.context.int_ptr


def test_programs_are_released_with_the_operator(device):
    handles = [run_solvers(device) for _ in range(3)]
    gc.collect()
    assert not set(handles) & cached_contexts()


def test_programs_are_reused_while_the_operator_lives(queue_and_operator):
    queue, R, data = queue_and_operator
    total_variation(R, data, mu=1.0, number_iterations=1, operator_norm=1.0)
    bundle = owner_leases(R)[("total_variation", queue.context)]
    total_variation(R, data, mu=1.0, number_iterations=1, operator_norm=1.0)
    assert owner_leases(R)[("total_variation", queue.context)] is bundle


def test_invalidation_rebuilds_the_programs(queue_and_operator):
    queue, R, data = queue_and_operator
    total_variation(R, data, mu=1.0, number_iterations=1, operator_norm=1.0)
    old = owner_leases(R)[("total_variation", queue.context)]
    invalidate_kernel_cache()
    total_variation(R, data, mu=1.0, number_iterations=1, operator_norm=1.0)
    assert owner_leases(R)[("total_variation", queue.context)] is not old


def test_weighted_dot_kernel_is_released_with_the_space(device):
    def use_space():
        queue = cl.CommandQueue(cl.Context([device]))
        X = Space((4, 3), axis_weights=(1, [1.0, 2.0, 3.0]))
        x = clarray.to_device(queue, np.ones((4, 3), np.float32))
        assert X.inner(x, x) == pytest.approx(24.0)
        assert owner_leases(X)
        return queue.context.int_ptr

    handle = use_space()
    gc.collect()
    assert handle not in cached_contexts()


def test_leases_are_not_copied_or_pickled(queue_and_operator):
    queue, R, data = queue_and_operator
    total_variation(R, data, mu=1.0, number_iterations=1, operator_norm=1.0)
    assert owner_leases(R)
    for copied in [copy.deepcopy(R), pickle.loads(pickle.dumps(R))]:
        assert "_kernel_leases" not in copied.__dict__
        assert "_kernel_leases" not in copied.codomain.__dict__


@pytest.fixture
def queue_and_operator(device):
    queue = cl.CommandQueue(cl.Context([device]))
    R = Radon(image_domain=8, angles=6)
    data = clarray.zeros(queue, R.output_shape, np.float32)
    return queue, R, data
