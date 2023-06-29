from functools import partial
from typing import Any, Tuple, Callable
from jaxtyping import Array
from jax.nn import softmax
from jax import numpy as jnp
from jax import random
from jax.lax import stop_gradient

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
    logits = jnp.log(jnp.stack([1 - rate, rate]))
    # sample gumbel noise
    gumbel = random.gumbel(key, logits.shape)
    # softmax -> probabilities for success and failure
    r = softmax((logits + gumbel) / temp, axis=0)
    # return most likely category
    y = jnp.argmax(r, axis=0)
    # attach the r success gradient
    y = stop_gradient(y - r[0]) + r[0]
    return y

def bernoulli(key: Any, rate: Array, shape: Tuple) -> Array:
    return random.bernoulli(key, rate, shape)
