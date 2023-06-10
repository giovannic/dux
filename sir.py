from typing import Any, Tuple
from dataclasses import dataclass
from jax import numpy as jnp
from jax import random, jit, grad
from jax.lax import scan
from jaxtyping import Array
from jax.tree_util import register_pytree_node

@dataclass
class State:
    susceptible: Array
    infected: Array
    recovered: Array

register_pytree_node(
    State,
    lambda tree: ((tree.susceptible, tree.infected, tree.recovered), None),
    lambda _, args: State(*args)
)

@dataclass
class Observation:
    n_susceptible: Array
    n_infected: Array
    n_recovered: Array

register_pytree_node(
    Observation,
    lambda tree: ((tree.n_susceptible, tree.n_infected, tree.n_recovered), None),
    lambda _, args: Observation(*args)
)

def init(infected: float, n: int, key: Any) -> State:
    n_infected = int(n * infected)
    return State(
        jnp.zeros((n,)).at[n_infected:].set(1),
        jnp.zeros((n,)).at[:n_infected].set(1),
        jnp.zeros((n,))
    )

def step(beta: float, gamma: float, state: State, key: Any) -> State:
    # get a population size
    n = state.susceptible.shape[0]

    # calculate force of infection
    foi = beta * jnp.sum(state.infected) / n

    # sample infections
    key, key_i = random.split(key)
    new_infections = random.bernoulli(key_i, state.susceptible * foi, (n,))

    # sample recoveries
    key, key_i = random.split(key)
    new_recoveries = random.bernoulli(key_i, state.infected * gamma, (n,))

    # make new state
    return State(
        state.susceptible - new_infections,
        state.infected - new_recoveries + new_infections,
        state.recovered + new_recoveries
    )

def observe(state: State) -> Observation:
    return Observation(
        state.susceptible.sum(),
        state.infected.sum(),
        state.recovered.sum()
    )

def _scan_step(
        beta: float,
        gamma: float,
        state: State,
        key: Any
    ) -> Tuple[State, Observation]:
    new_state = step(beta, gamma, state, key)
    return new_state, observe(new_state)

def run(
    n: int,
    infected: float,
    beta: float,
    gamma: float,
    timesteps: int,
    key: Any
    ) -> Observation:
    state = init(infected, n, key)
    _, obs = scan(
        f = lambda s, k: _scan_step(beta, gamma, s, k),
        init = state,
        xs = random.split(key, timesteps),
        length = timesteps
    )
    return obs

print(
    jit(run, static_argnums=[0, 1, 2, 3, 4])(
        1000, .5, .35, .25, 100, random.PRNGKey(0)
    )
)

def final_susceptible(*args) -> Array:
    obs = run(*args)
    return obs.n_susceptible[-1]

# This will obviously not work because bernoulli is not differentiable here
print(
    grad(final_susceptible, argnums=[2, 3])(
        1000, .5, .5, .25, 100, random.PRNGKey(0)
    )
)
