from tabulate import tabulate
from optax import score_function_jacobians, multi_normal
from typing import Tuple, Callable, Iterable
from jaxtyping import Array
from dataclasses import dataclass
from sir import final_susceptible
from jax import random, vmap, value_and_grad
from jax import numpy as jnp
from dux.bernoulli import BernoulliSig, bernoulli, make_gumbel_sm_approx
from functools import partial

def _many_value_and_grad(f: Callable, args: Array, n: int) -> Tuple:
    key = random.PRNGKey(0)
    keys = random.split(key, n)
    v, g = vmap(
        value_and_grad(f, argnums=1),
        in_axes=[0, None]
    )(keys, args)
    return v, g

def _run_gumbel_softmax(f: Callable, args: Array, n: int) -> Tuple:
    return _many_value_and_grad(
        partial(f, method=make_gumbel_sm_approx()),
        args,
        n
    )

def _run_bernoulli(f: Callable, args: Array, n: int) -> Tuple:
    return _many_value_and_grad(
        partial(f, method=bernoulli),
        args,
        n
    )

def _run_score_function(f: Callable, args: Array, n: int) -> Tuple:
    std = .1
    key = random.PRNGKey(0)
    key, key_i = random.split(key)

    def dist_builder(*args): 
        args = jnp.array(args)
        return multi_normal(args, jnp.log(jnp.full_like(args, std)))

    grads = score_function_jacobians(
        partial(f, key_i, method=bernoulli),
        args,
        dist_builder,
        key,
        n
    )

    keys = random.split(key, n)
    values = vmap(f, in_axes=[0, None])(keys, args)
    return values, grads 

@dataclass
class Method:
    name: str
    method: Callable

methods = [
    Method('Bernoulli', _run_bernoulli),
    Method('Gumbel Softmax', _run_gumbel_softmax),
    Method('Score function', _run_score_function)
]

def bernoulli_positive(key, p: Array, method: BernoulliSig = bernoulli) -> Array:
    shape = (10000,)
    y = jnp.sum(method(key, jnp.full(shape, p), shape)) / shape[0]
    return y

def run_final_susceptible(
    key,
    args: Array,
    method: BernoulliSig = bernoulli) -> Array:
    pop = 1000
    timesteps = 100
    n_susceptible = final_susceptible(
        key,
        pop,
        args[0],
        args[1],
        args[2],
        timesteps
    )
    return n_susceptible / pop

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

def summarise_run(values: Array, grads: Array) -> Tuple:
    value_entry = (jnp.mean(values), jnp.std(values))
    grad_entries = tuple(_flatten(
        zip(jnp.mean(grads, axis=0), jnp.std(grads, axis=0))
    ))

    return value_entry + grad_entries

print('-------------------')
print('Simple Bernoulli .3')
print('-------------------')

print(
    tabulate(
        [
            (method.name,) + summarise_run(
                *method.method(
                    bernoulli_positive, 
                    jnp.array([.3]),
                    10
                )
            )
            for method in methods
        ],
        headers = ['experiment', 'value', '(std)', 'Δp', '(std)']
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
            (method.name,) + summarise_run(
                *method.method(
                    run_final_susceptible,
                    jnp.array([.5, .25, .35]),
                    10
                )
            )
            for method in methods
        ],
        headers = ['experiment', 'value', '(std)'] + sir_arg_headers
    )
)
