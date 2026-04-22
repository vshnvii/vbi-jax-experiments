"""
Numerical integrators for JAX simulations.
"""
import jax.numpy as jnp

def euler_step(x, t, dt, rhs_fn, noise_fn, params, inputs, key):
    """
    Euler-Maruyama integration step.
    """
    dx = rhs_fn(x, t, params, inputs)
    noise = noise_fn(key, x.shape)
    return x + dt * dx + noise

def heun_step(x, t, dt, rhs_fn, noise_fn, params, inputs, key):
    """
    Heun formulation stochastic integration step.
    """
    noise = noise_fn(key, x.shape)
    dx1 = rhs_fn(x, t, params, inputs)
    x_aux = x + dt * dx1 + noise
    dx2 = rhs_fn(x_aux, t + dt, params, inputs)
    return x + 0.5 * dt * (dx1 + dx2) + noise
