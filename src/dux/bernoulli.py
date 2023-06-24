from functools import partial
from typing import Any, Tuple, Callable
from jaxtyping import Array
from jax.nn import softmax
from jax import numpy as jnp
from jax import random

BernoulliSig = Callable[[Any, Array, Tuple], Array]

def make_gumbel_sm_approx(
    temp=.1,
    min_rate=1e-10,
    max_rate=1.
    ):
    return partial(
        _gumbel_sm_approx,
        temp=temp,
        min_rate=min_rate,
        max_rate=max_rate
    )

def _gumbel_sm_approx(
    key: Any,
    rate: Array,
    shape: Tuple,
    temp=.1,
    min_rate=1e-10,
    max_rate=1.
    ) -> Array:
    # clip rates which are incompatible
    rate = jnp.minimum(jnp.maximum(rate, min_rate), max_rate)
    # stack rates for p and q
    logits = jnp.log(jnp.stack([rate, 1 - rate]))
    # sample gumbel noise
    gumbel = random.gumbel(key, shape)
    # softmax -> probabilities for success and failure
    r = softmax((logits + gumbel) / temp)
    # return probability of success
    return r[0]

def bernoulli(key: Any, rate: Array, shape: Tuple) -> Array:
    return random.bernoulli(key, rate, shape)
