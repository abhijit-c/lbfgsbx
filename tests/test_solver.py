from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import optimize

from lbfgsbx import LbfgsbResult, minimize


def rosenbrock(x):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


@pytest.mark.parametrize(
    ("x0", "lower", "upper"),
    [
        ((0.8, 1.2), (-1.0, -2.0), (2.0, 2.0)),
        ((2.0, 0.0), (1.5, -2.0), (3.0, 0.5)),
        ((2.0, 0.5), (0.0, -1.0), (3.0, 1.0)),
    ],
)
def test_rosenbrock_matches_scipy(x0, lower, upper):
    x0_array = jnp.asarray(x0)
    bounds = (jnp.asarray(lower), jnp.asarray(upper))

    result = minimize(
        rosenbrock,
        x0_array,
        bounds,
        maxiter=100,
        tol=1e-5,
        history_size=5,
    )
    reference = optimize.minimize(
        lambda x: np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2),
        np.asarray(x0),
        method="L-BFGS-B",
        bounds=list(zip(lower, upper)),
        options={"maxiter": 100, "maxcor": 5, "gtol": 1e-5},
    )

    assert isinstance(result, LbfgsbResult)
    assert result.success
    np.testing.assert_allclose(result.x, reference.x, atol=5e-5)
    assert bool(jnp.all(result.x >= bounds[0]))
    assert bool(jnp.all(result.x <= bounds[1]))


@pytest.mark.parametrize(
    ("lower", "upper"),
    [((0.0, -5.0, 0.0), (2.0, 0.0, 1.0)), ((0.0, 0.0, 0.0), (1.5, 1.0, 0.4))],
)
def test_bounded_quadratic_finds_projected_optimum(lower, upper):
    center = jnp.array([1.0, -2.0, 0.5])
    lower_array = jnp.asarray(lower)
    upper_array = jnp.asarray(upper)
    expected = jnp.clip(center, lower_array, upper_array)

    result = minimize(
        lambda x: jnp.sum((x - center) ** 2),
        jnp.zeros(3),
        (lower_array, upper_array),
        tol=1e-6,
        maxiter=20,
    )

    assert result.success
    np.testing.assert_allclose(result.x, expected, atol=1e-6)
    assert int(result.nit) <= 2
    assert int(result.nfev) == int(result.njev) == int(result.nls) + 1


@pytest.mark.parametrize(
    ("x0", "center"),
    [
        ((-3.0, -1.0, 4.0), (-2.0, 1.5, 0.5)),
        ((2.0, 3.0, -4.0), (1.0, -2.0, -0.5)),
    ],
)
def test_nonnegativity_constraints(x0, center):
    center_array = jnp.asarray(center)

    result = minimize(
        lambda x: jnp.sum((x - center_array) ** 2),
        jnp.asarray(x0),
        bounds=(0.0, jnp.inf),
        tol=1e-6,
    )

    assert result.success
    np.testing.assert_allclose(result.x, jnp.maximum(center_array, 0.0), atol=1e-6)
    assert bool(jnp.all(result.x >= 0.0))


def test_unconstrained_problem_and_args():
    target = jnp.array([2.0, -3.0])
    result = minimize(
        lambda x, center: jnp.sum((x - center) ** 2),
        jnp.zeros(2),
        args=(target,),
        tol=1e-6,
    )
    assert result.success
    np.testing.assert_allclose(result.x, target, atol=1e-6)


def test_pytree_and_scalar_bounds_match_flat_problem():
    params = {"a": jnp.array([0.0, 0.0]), "b": (jnp.array(0.0),)}
    target = {"a": jnp.array([1.0, -2.0]), "b": (jnp.array(0.5),)}

    def objective(tree):
        return jnp.sum((tree["a"] - target["a"]) ** 2) + (
            tree["b"][0] - target["b"][0]
        ) ** 2

    result = minimize(objective, params, bounds=(-1.0, 1.0), tol=1e-6)

    assert result.success
    np.testing.assert_allclose(result.x["a"], jnp.array([1.0, -1.0]), atol=1e-6)
    np.testing.assert_allclose(result.x["b"][0], 0.5, atol=1e-6)
    assert jax.tree.structure(result.x) == jax.tree.structure(params)
    assert jax.tree.structure(result.jac) == jax.tree.structure(params)


def test_fixed_variable_and_zero_gradient_component():
    result = minimize(
        lambda x: -x[0],
        jnp.array([-0.5, 0.25]),
        bounds=(jnp.array([-1.0, 0.25]), jnp.array([1.0, 0.25])),
        tol=1e-6,
    )
    assert result.success
    np.testing.assert_allclose(result.x, jnp.array([1.0, 0.25]), atol=1e-6)


def test_infeasible_initial_point_is_projected_before_evaluation():
    result = minimize(
        lambda x: jnp.sum((x - 0.5) ** 2),
        jnp.array([-10.0, 10.0]),
        bounds=(0.0, 1.0),
        tol=1e-6,
    )
    assert result.success
    np.testing.assert_allclose(result.x, jnp.array([0.5, 0.5]), atol=1e-6)


def test_array_like_bounds_are_accepted_for_array_parameters():
    result = minimize(
        lambda x: jnp.sum((x - 0.5) ** 2),
        jnp.zeros(2),
        bounds=([0.0, -1.0], [1.0, 0.25]),
        tol=1e-6,
    )
    assert result.success
    np.testing.assert_allclose(result.x, jnp.array([0.5, 0.25]), atol=1e-6)


def test_initial_optimum_and_zero_iterations():
    optimum = minimize(lambda x: jnp.sum(x**2), jnp.zeros(2), maxiter=0)
    exhausted = minimize(lambda x: jnp.sum((x - 1.0) ** 2), jnp.zeros(2), maxiter=0)

    assert optimum.success
    assert int(optimum.status) == 0
    assert int(optimum.nit) == 0
    assert not exhausted.success
    assert int(exhausted.status) == 1
    assert int(exhausted.nit) == 0


def test_jitted_pytree_solve():
    def objective(tree):
        return (tree["a"] - 1.0) ** 2 + (tree["b"] + 2.0) ** 2

    solve = jax.jit(
        lambda a, b: minimize(
            objective,
            {"a": a, "b": b},
            bounds=(-3.0, 3.0),
            tol=1e-6,
            maxiter=30,
        )
    )
    result = solve(jnp.array(0.0), jnp.array(0.0))

    assert result.success
    np.testing.assert_allclose(result.x["a"], 1.0, atol=1e-6)
    np.testing.assert_allclose(result.x["b"], -2.0, atol=1e-6)


def test_history_wraps_and_converges():
    result = minimize(
        rosenbrock,
        jnp.array([-1.2, 1.0]),
        bounds=(-3.0, 3.0),
        history_size=2,
        maxiter=200,
        tol=1e-5,
    )
    assert int(result.nit) > 2
    assert result.success
    np.testing.assert_allclose(result.x, jnp.ones(2), atol=2e-4)


def test_line_search_failure_status():
    result = minimize(
        lambda x: jnp.sum((x - 10.0) ** 4),
        jnp.zeros(2),
        maxiter=10,
        max_linesearch_steps=1,
    )
    assert not result.success
    assert int(result.status) == 2
    np.testing.assert_allclose(result.x, jnp.zeros(2))


def test_nonfinite_initial_value_status():
    result = minimize(lambda x: jnp.log(-x[0]), jnp.ones(1))
    assert not result.success
    assert int(result.status) == 3
    assert int(result.nit) == 0


def test_float64_dtype_is_preserved():
    with jax.enable_x64():
        x0 = jnp.zeros(2, dtype=jnp.float64)
        result = minimize(lambda x: jnp.sum((x - 1.0) ** 2), x0, tol=1e-10)
        assert result.x.dtype == jnp.float64
        assert result.jac.dtype == jnp.float64
        assert result.projected_gradient_norm.dtype == jnp.float64


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"maxiter": -1}, ValueError),
        ({"history_size": 0}, ValueError),
        ({"max_linesearch_steps": 0}, ValueError),
        ({"tol": -1.0}, ValueError),
        ({"args": []}, TypeError),
    ],
)
def test_invalid_options(kwargs, error):
    with pytest.raises(error):
        minimize(lambda x: jnp.sum(x**2), jnp.zeros(2), **kwargs)


def test_invalid_parameter_and_bound_inputs():
    with pytest.raises(TypeError, match="floating-point"):
        minimize(lambda x: jnp.sum(x), jnp.ones(2, dtype=jnp.int32))
    with jax.enable_x64(), pytest.raises(TypeError, match="same dtype"):
        minimize(
            lambda x: x[0].sum() + x[1].sum(),
            (jnp.ones(1, dtype=jnp.float32), jnp.ones(1, dtype=jnp.float64)),
        )
    with pytest.raises(ValueError, match="structure"):
        minimize(
            lambda x: jnp.sum(x["x"] ** 2),
            {"x": jnp.zeros(2)},
            bounds=({"wrong": 0.0}, {"wrong": 1.0}),
        )
    with pytest.raises(ValueError, match="must not exceed"):
        minimize(lambda x: jnp.sum(x**2), jnp.zeros(2), bounds=(1.0, -1.0))
