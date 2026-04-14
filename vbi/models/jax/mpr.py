"""
Montbrio-Pazo-Roxin (MPR) model in JAX (FIXED VERSION)
"""
from typing import NamedTuple
import jax.numpy as jnp
from .base import JaxNeuralMass
from .integrators import heun_step
from .noise import generate_noise


class MPRParams(NamedTuple):
    weights: jnp.ndarray
    tau: float = 1.0
    I: float = 0.0
    Delta: float = 1.0
    J: float = 15.0
    eta: float = -5.0
    cr: float = 1.0
    cv: float = 0.0


# ---------------------------
# FIXED RHS (NUMBA MATCHED)
# ---------------------------
def mpr_rhs(x, t, params: MPRParams, inputs):
    r = x[..., 0]
    V = x[..., 1]

    # Evaluate coupling natively inside RHS to catch intermediate solver shifts!
    I_c = params.cr * (params.weights @ r) + params.cv * (params.weights @ V)

    dr = (1 / params.tau) * (
        params.Delta / (jnp.pi * params.tau)
        + 2 * r * V
    )

    dV = (1 / params.tau) * (
        V**2
        + params.eta
        + params.J * params.tau * r
        + params.I
        + I_c
        - (jnp.pi**2) * (r**2) * (params.tau**2)
    )

    return jnp.stack([dr, dV], axis=-1)


# ---------------------------
# MODEL CLASS
# ---------------------------
class JaxMPRModel(JaxNeuralMass):

    def __init__(
        self,
        params: MPRParams,
        sigma: float = 0.037,
        dt: float = 0.1,
        integrator=heun_step
    ):
        super().__init__(params, dt, integrator)
        self.sigma = sigma

    def rhs(self, x, t, params, inputs):
        return mpr_rhs(x, t, params, inputs)

    def noise(self, key, shape):
        return generate_noise(key, shape, self.sigma, same_noise=False)

    def step(self, x, key, inputs):
        x_next = super().step(x, key, inputs)
        # Numba's heun_sde enforces r >= 0 at the end of each step
        r_next = jnp.maximum(x_next[..., 0], 0.0)
        return x_next.at[..., 0].set(r_next)
