# lbfgsbx

`lbfgsbx` is a small, standalone implementation of the L-BFGS-B algorithm for
JAX. It minimizes differentiable scalar functions subject to elementwise lower
and upper bounds, supports parameter pytrees, and can execute inside
`jax.jit`.

The core algorithm is implemented in this package. It uses Optax only for its
strong-Wolfe zoom line search and has no dependency on JAXopt.

## Installation

```bash
uv add git+ssh://git@github.com/abhijit-c/lbfgsbx
```

## Usage

```python
import jax.numpy as jnp

from lbfgsbx import minimize


def objective(x):
    return jnp.sum((x - jnp.array([1.0, -2.0, 0.5])) ** 2)


result = minimize(
    objective,
    x0=jnp.zeros(3),
    bounds=(jnp.array([0.0, -5.0, 0.0]), jnp.array([2.0, 0.0, 1.0])),
)

assert result.success
print(result.x)
```

Parameters may also be nested JAX pytrees. 
Each bound may be a matching pytree or a scalar applied to every parameter:

```python
params = {"weight": jnp.zeros(3), "bias": jnp.array(0.0)}

def pytree_objective(p):
    return jnp.sum((p["weight"] - 0.5) ** 2) + p["bias"] ** 2

result = minimize(pytree_objective, params, bounds=(-1.0, 1.0))
```

The public signature is:

```python
minimize(
    fun,
    x0,
    bounds=None,
    *,
    args=(),
    maxiter=100,
    tol=1e-5,
    history_size=10,
    max_linesearch_steps=30,
)
```

`fun` is invoked as `fun(x, *args)` and must return a scalar. Gradients are
computed with `jax.value_and_grad`. Infeasible initial parameters are projected
into the box before the first evaluation.

The returned `LbfgsbResult` contains the solution (`x`), objective and gradient
(`fun`, `jac`), projected-gradient norm, iteration/evaluation counts, and a
numeric status:

- `0`: converged
- `1`: maximum iterations reached
- `2`: line search failed to find a decreasing step
- `3`: a non-finite objective, gradient, or iterate was encountered

All result fields are JAX-compatible, so a solve can be returned from a
JIT-compiled function:

```python
import jax

jitted_solve = jax.jit(lambda x0: minimize(objective, x0, bounds=(-5.0, 5.0)))
result = jitted_solve(jnp.zeros(3))
```

Differentiating through the completed optimization routine, custom gradient
callbacks, auxiliary objective outputs, and warm-started Hessian histories are
not currently supported.

## Development

```bash
uv sync --dev
uv run pytest
```

## License

`lbfgsbx` is licensed under the Apache License 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE) for details and attribution.

## References

- This implementation was developed with reference to
  [JAXopt's L-BFGS-B implementation](https://github.com/google/jaxopt/blob/main/jaxopt/_src/lbfgsb.py),
  which is licensed under the
  [Apache License 2.0](https://github.com/google/jaxopt/blob/main/LICENSE).
- R. H. Byrd, P. Lu, J. Nocedal, and C. Zhu, “A Limited Memory Algorithm for
  Bound Constrained Optimization,” *SIAM Journal on Scientific Computing*,
  1995.
- J. Nocedal and S. J. Wright, *Numerical Optimization*, second edition,
  Springer, 2006.
