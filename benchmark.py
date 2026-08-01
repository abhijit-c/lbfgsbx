"""Benchmark lbfgsbx against SciPy's L-BFGS-B on Rosenbrock problems.

JAX compilation is excluded from the reported timings: every problem shape is
compiled and executed once before any timed repetition, and timed results are
explicitly synchronized with the accelerator/CPU backend.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize

from lbfgsbx import LbfgsbResult, minimize


jax.config.update("jax_enable_x64", True)


class BenchmarkResult(NamedTuple):
    dimension: int
    jax_seconds: float
    scipy_seconds: float
    jax_result: LbfgsbResult
    scipy_result: optimize.OptimizeResult


def rosenbrock_jax(x: jax.Array) -> jax.Array:
    """The n-dimensional Rosenbrock objective."""
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def initial_point(dimension: int) -> np.ndarray:
    """Return the conventional alternating Rosenbrock starting point."""
    x0 = np.full(dimension, -1.2, dtype=np.float64)
    x0[1::2] = 1.0
    return x0


def synchronize(tree: Any) -> None:
    """Wait until all JAX leaves have completed execution."""
    jax.tree.map(lambda leaf: leaf.block_until_ready(), tree)


def median_runtime(
    operation: Callable[[], Any],
    repeats: int,
    sync: Callable[[Any], None] | None = None,
) -> tuple[float, Any]:
    """Time an operation repeatedly and return its median and last result."""
    durations: list[float] = []
    result: Any = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = operation()
        if sync is not None:
            sync(result)
        durations.append(time.perf_counter() - start)
    return statistics.median(durations), result


def benchmark_dimension(
    dimension: int,
    *,
    repeats: int,
    maxiter: int,
    tol: float,
    history_size: int,
    max_linesearch_steps: int,
) -> BenchmarkResult:
    """Benchmark both solvers for one Rosenbrock dimensionality."""
    x0_numpy = initial_point(dimension)
    x0_jax = jnp.asarray(x0_numpy)

    # Keeping the objective closed over no dynamic Python state lets JAX cache
    # one executable for this dimension. The call below is compiled lazily.
    solve_jax = jax.jit(
        lambda x: minimize(
            rosenbrock_jax,
            x,
            bounds=(-2.0, 2.0),
            maxiter=maxiter,
            tol=tol,
            history_size=history_size,
            max_linesearch_steps=max_linesearch_steps,
        )
    )

    # Untimed warm-up: this performs tracing, compilation, and one full solve.
    # Synchronization is essential because JAX dispatch is asynchronous.
    warmup_result = solve_jax(x0_jax)
    synchronize(warmup_result)

    jax_seconds, jax_result = median_runtime(
        lambda: solve_jax(x0_jax), repeats, synchronize
    )

    scipy_bounds = optimize.Bounds(np.full(dimension, -2.0), np.full(dimension, 2.0))

    def solve_scipy() -> optimize.OptimizeResult:
        return optimize.minimize(
            optimize.rosen,
            x0_numpy.copy(),
            method="L-BFGS-B",
            jac=optimize.rosen_der,
            bounds=scipy_bounds,
            options={
                "maxiter": maxiter,
                "maxcor": history_size,
                "maxls": max_linesearch_steps,
                "gtol": tol,
                # Avoid making SciPy's function-change tolerance the primary
                # stopping condition; lbfgsbx uses projected-gradient norm.
                "ftol": np.finfo(np.float64).eps,
            },
        )

    scipy_seconds, scipy_result = median_runtime(solve_scipy, repeats)
    return BenchmarkResult(
        dimension=dimension,
        jax_seconds=jax_seconds,
        scipy_seconds=scipy_seconds,
        jax_result=jax_result,
        scipy_result=scipy_result,
    )


def print_results(results: Sequence[BenchmarkResult]) -> None:
    """Print a compact comparison table."""
    header = (
        f"{'n':>8}  {'lbfgsbx (s)':>12}  {'SciPy (s)':>10}  {'SciPy/JAX':>10}"
        f"  {'JAX nit':>8}  {'SciPy nit':>9}  {'JAX status':>10}  {'SciPy status':>12}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        ratio = result.scipy_seconds / result.jax_seconds
        print(
            f"{result.dimension:8d}  {result.jax_seconds:12.6f}"
            f"  {result.scipy_seconds:10.6f}  {ratio:10.3f}"
            f"  {int(result.jax_result.nit):8d}  {result.scipy_result.nit:9d}"
            f"  {int(result.jax_result.status):10d}  {result.scipy_result.status:12d}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=(10, 100, 1_000),
        help="problem sizes to benchmark (default: 10 100 1000)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="timed repetitions per solver and dimension (default: 5)",
    )
    parser.add_argument("--maxiter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--history-size", type=int, default=10)
    parser.add_argument("--max-linesearch-steps", type=int, default=30)
    args = parser.parse_args()

    if any(dimension < 2 for dimension in args.dimensions):
        parser.error("all dimensions must be at least 2")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.maxiter < 0:
        parser.error("--maxiter must be non-negative")
    if args.tol < 0:
        parser.error("--tol must be non-negative")
    if args.history_size < 1:
        parser.error("--history-size must be at least 1")
    if args.max_linesearch_steps < 1:
        parser.error("--max-linesearch-steps must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    print(
        f"JAX backend: {jax.default_backend()}; dtype: float64; "
        f"median of {args.repeats} timed runs"
    )
    print("JAX compilation and the first solve for each dimension are not timed.\n")

    results = []
    for dimension in args.dimensions:
        print(f"Warming up and benchmarking n={dimension}...", flush=True)
        results.append(
            benchmark_dimension(
                dimension,
                repeats=args.repeats,
                maxiter=args.maxiter,
                tol=args.tol,
                history_size=args.history_size,
                max_linesearch_steps=args.max_linesearch_steps,
            )
        )
    print()
    print_results(results)


if __name__ == "__main__":
    main()
