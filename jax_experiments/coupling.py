"""
Coupling functions for network models.
"""
import jax.numpy as jnp

def diffusive_coupling(x, weights):
    """
    Computes weighted inputs for local dynamics.
    Assumes `x` has shape (n_nodes, state_dim).
    Assumes `weights` has shape (n_nodes, n_nodes).
    Returns inputs of shape (n_nodes, state_dim).
    """
    return weights @ x

def sparse_diffusive_coupling(x, indices, weights):
    """
    Sparse connective handling (Issue #67).
    """
    # jnp.take(buffer, indices) * weights
    pass # To be fully defined iteratively
