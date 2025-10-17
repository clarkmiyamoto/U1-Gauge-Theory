import jax
import jax.numpy as jnp

from src.statistics import topo_charge

import pytest

def random_u1_links(key, shape):
    """Sample random U(1) link angles uniformly in [0, 2π)."""
    return 2 * jnp.pi * jax.random.uniform(key, shape)

@pytest.mark.parametrize("L", [4, 8, 16])
def test_topo_charge_integer(L, seed=0):
    """Check that the topological charge is approximately integer-valued."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)

    L = 8
    lattice_shape = (L,L)
    link_shape = (2,L,L)

    u1_ex1 = 2*jnp.pi*jax.random.uniform(keys[0], shape=link_shape)
    u1_ex2 = 2*jnp.pi*jax.random.uniform(keys[1], shape=link_shape)
    cfgs = jnp.stack((u1_ex1, u1_ex2), axis=0)
    Q = topo_charge(cfgs)
    # Assert Q is close to an integer
    assert jnp.allclose(Q, jnp.rint(Q), atol=1e-6), f'Topological charge not integer: {Q}'