import jax.numpy as jnp

def compute_u1_plaq(links, mu, nu):
    """Computes plaquette for 2D U(1) gauge theory.
    
    Args:
        links: jax array of shape (spacetime_dim, length_dim1, length_dim2)
        mu: first direction index
        nu: second direction index
    """
    # Plaquette angle: θ_mu(x) + θ_nu(x + e_mu) - θ_mu(x + e_nu) - θ_nu(x)
    return (
        links[:, mu]
        + jnp.roll(links[:, nu], shift=-1, axis=mu + 1)
        - jnp.roll(links[:, mu], shift=-1, axis=nu + 1)
        - links[:, nu]
    )