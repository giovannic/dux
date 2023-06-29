from typing import Tuple, Callable, Iterable, Union, Sequence
from jaxtyping import Array
from dataclasses import dataclass
from sir import final_susceptible
from jax import random, vmap, value_and_grad
from jax import numpy as jnp
from dux.bernoulli import BernoulliSig, bernoulli, make_gumbel_sm_approx
from functools import partial
from tabulate import tabulate

@dataclass
class Method:
    name: str
    method: Callable

methods = [
    Method('Bernoulli', bernoulli),
    Method('Gumbel Softmax', make_gumbel_sm_approx())
]

def run_bernoulli(method: BernoulliSig, n: int, p: float) -> Tuple:
    key = random.PRNGKey(0)
    keys = random.split(key, n)
    shape = (10000,)

    def f_n_positive(k, p):
        return 1. * jnp.sum(method(k, jnp.full(shape, p), shape))

    v, g = vmap(
        value_and_grad(f_n_positive, argnums=1),
        in_axes=[0, None]
    )(keys, p)
    return v, (g,)

def run_sir(method: BernoulliSig, n: int) -> Tuple:
    """
    Returns values and gradients for an SIR run
    """
    key = random.PRNGKey(0)
    keys = random.split(key, n)
    fs_with_method = partial(final_susceptible, bernoulli=method)

    return vmap(
        value_and_grad(fs_with_method, argnums=[2, 3, 4]),
        in_axes=[0, None, None, None, None, None]
    )(keys, 1000, .5, .35, .25, 100)

def _flatten(xs: Iterable) -> Iterable:
    return (y for ys in xs for y in ys)

def summarise_run(values: Array, grads: Tuple) -> Tuple:
    return (
        jnp.mean(values),
        jnp.std(values),
    ) + tuple(_flatten((jnp.mean(g), jnp.std(g)) for g in grads))

print('-------------------')
print('Simple Bernoulli .3')
print('-------------------')

print(
    tabulate(
        [
            (method.name,) + summarise_run(
                *run_bernoulli(method.method, 10, .3)
            )
            for method in methods
        ],
        headers = ['value', '(std)', 'Δp', '(std)']
    )
)

print('-------------------')
print('Simple Bernoulli .7')
print('-------------------')

print(
    tabulate(
        [
            (method.name,) + summarise_run(
                *run_bernoulli(method.method, 10, .7)
            )
            for method in methods
        ],
        headers = ['value', '(std)', 'Δp', '(std)']
    )
)

print('---')
print('SIR')
print('---')

sir_args = ['infected', 'beta', 'gamma']
sir_arg_headers = list(_flatten((f'Δ{a}', '(std)') for a in sir_args))

print(
    tabulate(
        [
            (method.name,) + summarise_run(*run_sir(method.method, 10))
            for method in methods
        ],
        headers = ['value', '(std)'] + sir_arg_headers
    )
)
