from typing import Any, Tuple
from dataclasses import dataclass
from jax import numpy as jnp
from jax import random
from jax.lax import scan
from jaxtyping import Array
from jax.tree_util import register_pytree_node
from dux.bernoulli import BernoulliSig, bernoulli
import networkx as nx

@dataclass
class State:
    """
    :param susceptible: array of binary indicators showing whether an individual is susceptible
    :param infected: array of binary indicators showing whether an individual is infected
    :param recovered: array of binary indicators showing whether an individual is recovered
    """
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
    """
    :param n_susceptible: total number of susceptible individuals
    :param n_infected: total number of infected individuals
    :param n_recovered: total number of recovered individuals
    """
    n_susceptible: Array
    n_infected: Array
    n_recovered: Array

register_pytree_node(
    Observation,
    lambda tree: ((tree.n_susceptible, tree.n_infected, tree.n_recovered), None),
    lambda _, args: Observation(*args)
)

def init(initial_infected: float, n: int) -> State:
    """
    :param initial_infected: initial fraction of infected individuals
    :param n: population size
    :param key: a PRNG key used as the random key
    """
    n_infected = jnp.array(n * initial_infected, int)
    susceptible = (jnp.arange(n) < n_infected).astype(int)
    return State(
        susceptible,
        1 - susceptible,
        jnp.zeros((n,))
    )

def step(
    key: Any,
    beta: float,
    gamma: float,
    state: State,
    bernoulli: BernoulliSig,
    graph: nx.Graph
    ) -> State:
    """
    Runs the model forward for one timestep.

    **Arguments**:

    :param rng_key: a PRNG key used as the random key.
    :param beta: 
    :param gamma:
    :param state:
    :param bernoulli: 
    :param graph: a networkx graph
    """

    # convert graph from networkx to jnp adjacency-matrix
    A = nx.to_numpy_array(graph, weight=None)
    A = jnp.array(A)

    # get population size
    n = state.susceptible.shape[0]

    # calculate force of infection based on adjacency
    foi = beta * jnp.sum(jnp.matmul(A, state.infected)) / n

    # sample infections
    key, key_i = random.split(key)
    new_infections = bernoulli(key_i, state.susceptible * foi, (n,))

    # sample recoveries
    key, key_i = random.split(key)
    new_recoveries = bernoulli(key_i, state.infected * gamma, (n,))

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
        key: Any,
        beta: float,
        gamma: float,
        state: State,
        bernoulli: BernoulliSig,
        graph: nx.Graph
    ) -> Tuple[State, Observation]:
    new_state = step(key, beta, gamma, state, bernoulli, graph)
    return new_state, observe(new_state)

def run(
    key: Any,
    n: int,
    initial_infected: float,
    beta: float,
    gamma: float,
    timesteps: int,
    graph: nx.Graph,
    bernoulli: BernoulliSig=bernoulli
    ) -> Observation:
    state = init(initial_infected, n)
    _, obs = scan(
        f = lambda s, k: _scan_step(k, beta, gamma, s, graph, bernoulli),
        init = state,
        xs = random.split(key, timesteps),
        length = timesteps
    )
    return obs

def final_susceptible(*args, **kwargs) -> Array:
    obs = run(*args, **kwargs)
    return obs.n_susceptible[-1]

# key =  random.PRNGKey(1)
# n = 5
# initial_infected = 0.4
# state = init(initial_infected, n)
# graph=nx.gnp_random_graph(n, p=0.3)