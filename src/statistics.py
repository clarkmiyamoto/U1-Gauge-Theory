'''
Interesting statistics for U(1) gauge theory.
Implemented from https://arxiv.org/pdf/2101.08176.

- `topo_charge`: Topological charge. 
    In MCMC time, this quantity tends to freeze (or update very slowly).
    It's interesting to construct new MCMC methods which overcome this issue.
'''

import jax.numpy as jnp
from src.utils import compute_u1_plaq

def jax_wrap(x):
    return jnp.remainder(x+jnp.pi, 2*jnp.pi) - jnp.pi

def topo_charge(x):
    P01 = jax_wrap(compute_u1_plaq(x, mu=0, nu=1))
    # Sum over all spatial axes (exclude direction axis which was removed in utils)
    Q = jnp.sum(P01, axis=(-2, -1)) / (2 * jnp.pi)
    return Q