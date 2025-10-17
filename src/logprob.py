import jax
import jax.numpy as jnp

from src.utils import compute_u1_plaq

class U1GaugeAction:
    def __init__(self, 
                 beta: float, 
                 VolShape: tuple[int, ...],
                 ):
        '''
        Log Probability of 2D U(1) gauge theory with Wilson action

        Args:
            beta: Wilson β
            VolShape: Volume shape of the lattice. Example (8, 8) for 2D 8x8 lattice
        '''
        self.beta = beta
        self.VolShape = VolShape
        self.Nd = 2 # Number of spacetime dimensions

    def __call__(self, cfgs):
        '''
        Args:
            cfgs (jnp.ndarray): This is the plaquette angles \theta_{\mu \nu}(\vec n).
                The array is indexed via (\mu, \nu, x, y). 
                Since this is 2D, \mu and \nu are in {0,1},
                and \vec n = (x,y) are the spatial dimensions, they are in VolShape.
        '''
        Nd = cfgs.shape[1]
        # Accumulate Wilson action from plaquettes with mu < nu
        action_density = 0.0
        for mu in range(Nd):
            for nu in range(mu + 1, Nd):
                plaq = compute_u1_plaq(cfgs, mu, nu)
                action_density = action_density + jnp.cos(plaq)

        # Sum over all lattice sites
        return -self.beta * jnp.sum(action_density, axis=tuple(range(1,Nd+1)))

def build_u1_logprob(beta, VolShape):
    """Build a U(1) log probability function for given shape and beta"""
    u1_action = U1GaugeAction(beta, VolShape)
    
    def logprob(cfg):
        '''
        Args:
            cfg (jnp.ndarray): Array shaped (2, *VolShape).
                This is the link angles \theta_{\mu}(\vec n).
                The array is indexed via (\mu, x, y). 
                Since this is 2D, \mu is in {0,1},
                and \vec n = (x,y) are the spatial dimensions, they are in VolShape.
        '''
        shape = (2,) + VolShape
        cfg = cfg.reshape(shape)
        return u1_action(cfg[jnp.newaxis, ...])[0]
    
    return logprob, u1_action