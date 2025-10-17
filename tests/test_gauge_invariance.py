import jax
import jax.numpy as jnp

import pytest

def gauge_transform(links, alpha):
    """Apply gauge transformation to links"""

    for mu in range(len(links.shape[2:])):
        links = links.at[:,mu].set( alpha + links[:,mu] - jnp.roll(alpha, -1, mu+1))
    return links

def random_gauge_transform(x, key):
    """Apply random gauge transformation"""
    Nconf, VolShape = x.shape[0], x.shape[2:]
    
    # Generate random alpha
    alpha = 2 * jnp.pi * jax.random.uniform(key, (Nconf,) + VolShape)
    
    return gauge_transform(x, alpha)

@pytest.mark.parametrize("seed", [1, 2, 3,])
def test_invariance(seed, beta=2.0, tol=1e-8):
    '''Make sure actions invariant under gauge transformations'''
    from src.logprob import U1GaugeAction

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)

    L = 8
    lattice_shape = (L,L)
    link_shape = (2,L,L)

    u1_ex1 = 2*jnp.pi*jax.random.uniform(keys[0], shape=link_shape)
    u1_ex2 = 2*jnp.pi*jax.random.uniform(keys[1], shape=link_shape)
    cfgs = jnp.stack((u1_ex1, u1_ex2), axis=0)
    print(cfgs)
    

    u1_action = U1GaugeAction(beta=beta, VolShape=lattice_shape,)
    
    cfgs_transformed = random_gauge_transform(cfgs, keys[2])
    assert jnp.allclose(u1_action(cfgs), u1_action(cfgs_transformed), atol=tol)

if __name__ == "__main__":
    seed = 1
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)
    L = 8
    lattice_shape = (L,L)
    link_shape = (2,L,L)

    u1_ex1 = 2*jnp.pi*jax.random.uniform(keys[0], shape=link_shape)
    u1_ex2 = 2*jnp.pi*jax.random.uniform(keys[1], shape=link_shape)
    cfgs = jnp.stack((u1_ex1, u1_ex2), axis=0)

    print(cfgs.shape)