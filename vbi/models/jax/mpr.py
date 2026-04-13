"""
Montbrio-Pazo-Roxin (MPR) model in JAX.
"""
from typing import NamedTuple
import jax.numpy as jnp
from .base import JaxNeuralMass
from .integrators import euler_step
from .noise import generate_noise

class MPRParams(NamedTuple):
    tau: float = 1.0
    I: float = 0.0
    Delta: float = 1.0
    J: float = 15.0
    eta: float = -5.0
    cr: float = 1.0
    cv: float = 0.0

def mpr_rhs(x, t, params: MPRParams, inputs):
    # x: (n_nodes, 2) where x[..., 0] is r and x[..., 1] is V
    # inputs: (n_nodes, 2)
    r = x[..., 0]
    V = x[..., 1]
    
    # bound rate to be positive
    r = r * (r > 0)
    
    # compute input components
    I_c = params.cr * inputs[..., 0] + params.cv * inputs[..., 1]
    
    dr = (1 / params.tau) * (params.Delta / (jnp.pi * params.tau) + 2 * r * V)
    dV = (1 / params.tau) * (V ** 2 + params.eta + params.J * params.tau * r + params.I + I_c - (jnp.pi ** 2) * (r ** 2) * (params.tau ** 2))
    
    return jnp.stack([dr, dV], axis=-1)

class JaxMPRModel(JaxNeuralMass):
    def __init__(self, params: MPRParams = MPRParams(), sigma: float = 0.0, dt: float = 0.1, integrator=euler_step):
        super().__init__(params, dt, integrator)
        self.sigma = sigma

    def rhs(self, x, t, params, inputs):
        return mpr_rhs(x, t, params, inputs)

    def noise(self, key, shape):
        return generate_noise(key, shape, self.sigma, same_noise=False)
