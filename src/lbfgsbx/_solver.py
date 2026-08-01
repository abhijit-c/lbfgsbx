# Copyright 2023 Google LLC
# Modifications Copyright 2026 Abhijit Chowdhary
#
# This file contains code derived from JAXopt's L-BFGS-B implementation and
# has been substantially modified for the lbfgsbx project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Limited-memory BFGS with box constraints.

The implementation follows Byrd, Lu, Nocedal, and Zhu (1995): it computes a
generalized Cauchy point, minimizes the compact L-BFGS model over the free
variables, and globalizes the step with a strong-Wolfe line search.
"""

from __future__ import annotations

import numbers
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax.flatten_util import ravel_pytree


class LbfgsbResult(NamedTuple):
    """Result returned by :func:`minimize`.

    Status codes are ``0`` for convergence, ``1`` for reaching ``maxiter``,
    ``2`` for a failed line search, and ``3`` for non-finite numerical values.
    Scalar fields are JAX arrays so the result can be returned from ``jax.jit``.
    """

    x: PyTree
    fun: jax.Array
    jac: PyTree
    projected_gradient_norm: jax.Array
    nit: jax.Array
    nfev: jax.Array
    njev: jax.Array
    nls: jax.Array
    success: jax.Array
    status: jax.Array


class _State(NamedTuple):
    x: jax.Array
    value: jax.Array
    grad: jax.Array
    error: jax.Array
    s_history: jax.Array
    y_history: jax.Array
    theta: jax.Array
    num_updates: jax.Array
    linesearch_state: Any
    nit: jax.Array
    nfev: jax.Array
    njev: jax.Array
    nls: jax.Array
    linesearch_failed: jax.Array
    numerical_error: jax.Array


def _validate_options(
    maxiter: int, tol: float, history_size: int, max_linesearch_steps: int
) -> None:
    for name, value, minimum in (
        ("maxiter", maxiter, 0),
        ("history_size", history_size, 1),
        ("max_linesearch_steps", max_linesearch_steps, 1),
    ):
        if not isinstance(value, numbers.Integral) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    if not isinstance(tol, numbers.Real) or isinstance(tol, bool):
        raise TypeError("tol must be a real scalar")
    if tol < 0:
        raise ValueError("tol must be non-negative")


def _validate_params(x0) -> tuple[list[jax.Array], jnp.dtype]:
    leaves = jax.tree.leaves(x0)
    if not leaves:
        raise ValueError("x0 must contain at least one array leaf")

    arrays = [jnp.asarray(leaf) for leaf in leaves]
    if any(array.size == 0 for array in arrays):
        raise ValueError("x0 leaves must be non-empty")
    if any(not jnp.issubdtype(array.dtype, jnp.floating) for array in arrays):
        raise TypeError("x0 leaves must have floating-point dtypes")
    dtype = arrays[0].dtype
    if any(array.dtype != dtype for array in arrays[1:]):
        raise TypeError("all x0 leaves must have the same dtype")
    return arrays, dtype


def _broadcast_bound(bound, x0, dtype: jnp.dtype, name: str):
    x_structure = jax.tree.structure(x0)
    bound_structure = jax.tree.structure(bound)
    array_parameter = jax.tree_util.treedef_is_leaf(x_structure)

    if array_parameter:
        # An array parameter is one pytree leaf, so treat ordinary Python
        # sequences as array-like bounds rather than as nested containers.
        bound_tree = bound
    elif bound_structure == x_structure:
        bound_tree = bound
    elif jax.tree_util.treedef_is_leaf(bound_structure):
        bound_tree = jax.tree.map(lambda _: bound, x0)
    else:
        raise ValueError(f"{name} must be a scalar or have the same pytree structure as x0")

    def broadcast(value: Any, x: Any) -> jax.Array:
        try:
            return jnp.broadcast_to(jnp.asarray(value, dtype=dtype), jnp.shape(x))
        except ValueError as error:
            raise ValueError(
                f"each {name} leaf must be broadcastable to its x0 leaf"
            ) from error

    if array_parameter:
        return broadcast(bound_tree, x0)
    return jax.tree.map(broadcast, bound_tree, x0)


def _prepare_bounds(bounds, x0, dtype: jnp.dtype):
    if bounds is None:
        lower = jax.tree.map(lambda x: jnp.full_like(x, -jnp.inf), x0)
        upper = jax.tree.map(lambda x: jnp.full_like(x, jnp.inf), x0)
        return lower, upper
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        raise TypeError("bounds must be None or a (lower, upper) tuple")

    lower = _broadcast_bound(bounds[0], x0, dtype, "lower bound")
    upper = _broadcast_bound(bounds[1], x0, dtype, "upper bound")

    # Preserve tracing support: validate values only when concrete arrays are
    # available. Structure and broadcastability are always validated above.
    try:
        invalid = any(
            bool(jnp.any(lo > hi))
            for lo, hi in zip(jax.tree.leaves(lower), jax.tree.leaves(upper))
        )
    except (jax.errors.TracerBoolConversionError, TypeError):
        invalid = False
    if invalid:
        raise ValueError("lower bounds must not exceed upper bounds")
    return lower, upper


def _ravel(tree) -> jax.Array:
    leaves = jax.tree.leaves(tree)
    return jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])


def _projected_gradient_norm(
    x: jax.Array, grad: jax.Array, lower: jax.Array, upper: jax.Array
) -> jax.Array:
    residual = jnp.clip(x - grad, lower, upper) - x
    return jnp.max(jnp.abs(residual))


def _chronological_history(
    history: jax.Array, num_updates: jax.Array
) -> tuple[jax.Array, jax.Array]:
    size = history.shape[0]
    count = jnp.minimum(num_updates, size)
    start = jnp.where(num_updates < size, 0, num_updates % size)
    indices = (start + jnp.arange(size)) % size
    valid = jnp.arange(size) < count
    ordered = history[indices] * valid[:, None].astype(history.dtype)
    return ordered, valid


def _compact_matrices(
    s_history: jax.Array,
    y_history: jax.Array,
    theta: jax.Array,
    num_updates: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Construct W and M from equations (3.3)--(3.4) of the paper."""
    s, valid = _chronological_history(s_history, num_updates)
    y, _ = _chronological_history(y_history, num_updates)
    size = s.shape[0]

    w = jnp.concatenate((y.T, theta * s.T), axis=1)
    syt = s @ y.T
    sst = s @ s.T
    lower = jnp.tril(syt, -1)

    count = jnp.minimum(num_updates, size)
    last = jnp.maximum(count - 1, 0)
    pad = (~valid).astype(s.dtype)
    syt_scale = jnp.where(count > 0, syt[last, last], jnp.asarray(1, s.dtype))
    sst_scale = jnp.where(count > 0, theta * sst[last, last], jnp.asarray(1, s.dtype))
    syt_scale = jnp.maximum(jnp.abs(syt_scale), jnp.finfo(s.dtype).tiny)
    sst_scale = jnp.maximum(jnp.abs(sst_scale), jnp.finfo(s.dtype).tiny)

    diagonal = -jnp.diag(jnp.diag(syt)) + jnp.diag(pad * syt_scale)
    bottom_right = theta * sst + jnp.diag(pad * sst_scale)
    m_inverse = jnp.block(
        [[diagonal, lower.T], [lower, bottom_right]]
    )
    return w, jnp.linalg.inv(m_inverse)


def _generalized_cauchy_point(
    x: jax.Array,
    grad: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    theta: jax.Array,
    w: jax.Array,
    m: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Find the first local model minimizer along projected steepest descent."""
    eps = jnp.finfo(x.dtype).eps
    breakpoints = jnp.where(
        jnp.abs(grad) < eps,
        jnp.inf,
        jnp.where(grad < 0, (x - upper) / grad, (x - lower) / grad),
    )
    direction = jnp.where(breakpoints < eps, 0, -grad)
    bound_point = jnp.where(
        direction > 0, upper, jnp.where(direction < 0, lower, x)
    )

    permutation = jnp.argsort(breakpoints)
    sorted_breakpoints = breakpoints[permutation]
    intervals = jnp.diff(jnp.pad(sorted_breakpoints, (1, 0)))
    free_sorted = sorted_breakpoints > eps
    start = jnp.argmax(jnp.concatenate((free_sorted, jnp.ones(1, dtype=bool))))

    c0 = jnp.zeros(m.shape[-1], dtype=m.dtype)
    p0 = w.T @ direction
    derivative0 = -(direction @ direction)
    second_derivative0 = -theta * derivative0 - p0 @ (m @ p0)
    initial = (start, derivative0, second_derivative0, free_sorted, c0, p0)

    def condition(state: tuple[Any, ...]) -> jax.Array:
        i, derivative, second_derivative, *_ = state
        return (i < x.size) & (-derivative / second_derivative >= intervals[i])

    def body(state: tuple[Any, ...]) -> tuple[Any, ...]:
        i, derivative, second_derivative, free, c, p = state
        original_index = permutation[i]
        new_c = c + intervals[i] * p
        new_derivative = (
            derivative
            + intervals[i] * second_derivative
            + grad[original_index] ** 2
            + theta * grad[original_index] * (bound_point[original_index] - x[original_index])
            - grad[original_index] * w[original_index] @ (m @ new_c)
        )
        new_second_derivative = (
            second_derivative
            - theta * grad[original_index] ** 2
            - 2 * grad[original_index] * w[original_index] @ (m @ p)
            - grad[original_index] ** 2 * w[original_index] @ (m @ w[original_index])
        )
        new_second_derivative = jnp.maximum(eps, new_second_derivative)
        new_p = p + grad[original_index] * w[original_index]
        return (
            i + 1,
            new_derivative,
            new_second_derivative,
            free.at[i].set(False),
            new_c,
            new_p,
        )

    i, derivative, second_derivative, free_sorted, c, p = jax.lax.while_loop(
        condition, body, initial
    )
    step = jnp.maximum(-derivative / second_derivative, 0)
    step = jnp.where(jnp.isfinite(step), step, 0)
    elapsed = jax.lax.cond(i > 0, lambda: sorted_breakpoints[i - 1], lambda: jnp.asarray(0, x.dtype))
    elapsed = elapsed + step
    free = free_sorted[jnp.argsort(permutation)]
    cauchy = jnp.where(free, x + elapsed * direction, bound_point)
    return cauchy, c + step * p, free


def _subspace_minimum(
    x: jax.Array,
    grad: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    cauchy: jax.Array,
    c: jax.Array,
    theta: jax.Array,
    w: jax.Array,
    m: jax.Array,
    free: jax.Array,
) -> jax.Array:
    """Minimize the quadratic model over the Cauchy point's free variables."""
    masked_w = w * free[:, None].astype(w.dtype)
    residual = grad + theta * (cauchy - x) - masked_w @ (m @ c)

    vector = m @ (masked_w.T @ residual)
    system = jnp.eye(m.shape[0], dtype=m.dtype) - m @ (masked_w.T @ masked_w) / theta
    vector = jnp.linalg.solve(system, vector)
    step = -residual / theta - masked_w @ vector / theta**2

    ratios = jnp.maximum((upper - cauchy) / step, (lower - cauchy) / step)
    ratios = jnp.where(free & (jnp.abs(step) > 0), ratios, 1)
    scale = jnp.minimum(jnp.min(ratios), 1)
    return jnp.where(free, cauchy + scale * step, cauchy)


def minimize(
    fun: Callable[..., jax.Array],
    x0,
    bounds=None,
    *,
    args: tuple[Any, ...] = (),
    maxiter: int = 100,
    tol: float = 1e-5,
    history_size: int = 10,
    max_linesearch_steps: int = 30,
) -> LbfgsbResult:
    """Minimize a differentiable scalar function subject to box constraints.

    ``fun`` is called as ``fun(x, *args)``. ``x0`` may be an array or a pytree
    of arrays. Bounds are a ``(lower, upper)`` tuple whose members are either
    scalars or pytrees matching ``x0``.

    The optimization loop is compatible with ``jax.jit``, but differentiation
    through the returned solution is not supported.
    """
    if not callable(fun):
        raise TypeError("fun must be callable")
    if not isinstance(args, tuple):
        raise TypeError("args must be a tuple")
    _validate_options(maxiter, tol, history_size, max_linesearch_steps)
    _, dtype = _validate_params(x0)
    lower_tree, upper_tree = _prepare_bounds(bounds, x0, dtype)

    flat_x0, unravel = ravel_pytree(x0)
    lower = _ravel(lower_tree)
    upper = _ravel(upper_tree)
    initial_x = jnp.clip(flat_x0, lower, upper)

    def flat_fun(flat_x: jax.Array) -> jax.Array:
        return fun(unravel(flat_x), *args)

    value_and_grad = jax.value_and_grad(flat_fun)
    initial_value, initial_grad = value_and_grad(initial_x)
    initial_error = _projected_gradient_norm(initial_x, initial_grad, lower, upper)
    initial_numerical_error = ~(
        jnp.isfinite(initial_value) & jnp.all(jnp.isfinite(initial_grad))
    )

    linesearch = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps,
        max_learning_rate=1.0,
        initial_guess_strategy="one",
    )
    linesearch_state = linesearch.init(initial_x)
    state = _State(
        x=initial_x,
        value=initial_value,
        grad=initial_grad,
        error=initial_error,
        s_history=jnp.zeros((history_size, initial_x.size), dtype=dtype),
        y_history=jnp.zeros((history_size, initial_x.size), dtype=dtype),
        theta=jnp.asarray(1, dtype=dtype),
        num_updates=jnp.asarray(0, dtype=jnp.int32),
        linesearch_state=linesearch_state,
        nit=jnp.asarray(0, dtype=jnp.int32),
        nfev=jnp.asarray(1, dtype=jnp.int32),
        njev=jnp.asarray(1, dtype=jnp.int32),
        nls=jnp.asarray(0, dtype=jnp.int32),
        linesearch_failed=jnp.asarray(False),
        numerical_error=initial_numerical_error,
    )

    def condition(current: _State) -> jax.Array:
        return (
            (current.nit < maxiter)
            & (current.error > tol)
            & ~current.linesearch_failed
            & ~current.numerical_error
        )

    def body(current: _State) -> _State:
        w, m = _compact_matrices(
            current.s_history,
            current.y_history,
            current.theta,
            current.num_updates,
        )
        cauchy, c, free = _generalized_cauchy_point(
            current.x, current.grad, lower, upper, current.theta, w, m
        )
        target = _subspace_minimum(
            current.x,
            current.grad,
            lower,
            upper,
            cauchy,
            c,
            current.theta,
            w,
            m,
            free,
        )
        direction = target - current.x
        updates, new_linesearch_state = linesearch.update(
            direction,
            current.linesearch_state,
            current.x,
            value=current.value,
            grad=current.grad,
            value_fn=flat_fun,
        )
        candidate_x = current.x + updates
        candidate_value = new_linesearch_state.value
        candidate_grad = new_linesearch_state.grad
        line_steps = new_linesearch_state.info.num_linesearch_steps.astype(jnp.int32)
        numerical_error = ~(
            jnp.isfinite(candidate_value)
            & jnp.all(jnp.isfinite(candidate_grad))
            & jnp.all(jnp.isfinite(candidate_x))
        )
        line_failed = (
            (new_linesearch_state.info.decrease_error > 0)
            | (new_linesearch_state.learning_rate <= 0)
        ) & ~numerical_error
        accepted = ~(numerical_error | line_failed)

        step = candidate_x - current.x
        grad_step = candidate_grad - current.grad
        curvature = step @ grad_step
        curvature_floor = (
            jnp.finfo(dtype).eps * jnp.linalg.norm(step) * jnp.linalg.norm(grad_step)
        )
        update_history = accepted & jnp.isfinite(curvature) & (curvature > curvature_floor)
        history_index = current.num_updates % history_size
        new_s_history = jax.lax.cond(
            update_history,
            lambda history: history.at[history_index].set(step),
            lambda history: history,
            current.s_history,
        )
        new_y_history = jax.lax.cond(
            update_history,
            lambda history: history.at[history_index].set(grad_step),
            lambda history: history,
            current.y_history,
        )
        new_theta = jnp.where(
            update_history, (grad_step @ grad_step) / curvature, current.theta
        )
        new_num_updates = current.num_updates + update_history.astype(jnp.int32)

        new_x = jnp.where(accepted, candidate_x, current.x)
        new_value = jnp.where(accepted, candidate_value, current.value)
        new_grad = jnp.where(accepted, candidate_grad, current.grad)
        new_error = _projected_gradient_norm(new_x, new_grad, lower, upper)
        return _State(
            x=new_x,
            value=new_value,
            grad=new_grad,
            error=new_error,
            s_history=new_s_history,
            y_history=new_y_history,
            theta=new_theta,
            num_updates=new_num_updates,
            linesearch_state=new_linesearch_state,
            nit=current.nit + 1,
            nfev=current.nfev + line_steps,
            njev=current.njev + line_steps,
            nls=current.nls + line_steps,
            linesearch_failed=line_failed,
            numerical_error=numerical_error,
        )

    final = jax.lax.while_loop(condition, body, state)
    converged = (final.error <= tol) & ~final.numerical_error
    status = jnp.where(
        final.numerical_error,
        3,
        jnp.where(converged, 0, jnp.where(final.linesearch_failed, 2, 1)),
    ).astype(jnp.int32)
    return LbfgsbResult(
        x=unravel(final.x),
        fun=final.value,
        jac=unravel(final.grad),
        projected_gradient_norm=final.error,
        nit=final.nit,
        nfev=final.nfev,
        njev=final.njev,
        nls=final.nls,
        success=converged,
        status=status,
    )
