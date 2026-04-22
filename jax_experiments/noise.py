"""
Noise models for JAX simulations.
"""
import jax.numpy as jnp
from jax import random

def generate_noise(key, shape, sigma, same_noise=False):
    """
    Generates noise for integration steps.
    """
    if same_noise:
        z = random.normal(key, (shape[-1],))
        return jnp.broadcast_to(z, shape) * sigma
    else:
        return random.normal(key, shape) * sigma
