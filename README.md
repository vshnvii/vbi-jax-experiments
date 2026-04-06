# JAX Simulation Framework 

This is my initial attempt at building a simple JAX-based simulation setup for Virtual Brain Inference (VBI), mainly to understand the design discussed in Issue #67.

## Overview

The goal here was to try out a minimal JAX implementation that follows a more functional approach (instead of heavy class-based design), so that it works well with `jax.jit` and `jax.vmap`.

I started with a basic neural mass model (MPR) and built a small pipeline around it to simulate multiple nodes.

## What’s included

- **`base.py`**  
  Contains a simple base structure (`JaxNeuralMass`) where the model equations (RHS) are kept separate from the simulation loop (using `lax.scan`).

- **`integrators.py`**  
  Basic numerical integration methods like Euler and Heun implemented in a functional way.

- **`mpr.py`**  
  Implementation of the MPR neural mass model. Parameters are defined using `NamedTuple` so they work nicely with JAX transformations.

- **`coupling.py`**  
  Simple network coupling (like `W @ x`) to connect multiple nodes.

- **`noise.py`**  
  Noise generation using JAX PRNG keys, keeping things reproducible.

## Setup

Make sure JAX is installed:

```bash
pip install jax jaxlib

```bash
# CPU Version
pip install -e ".[jax]"

# GPU acceleration (Optional)
pip install -U "jax[cuda12_pip]"
```

## Quick Start Example

You can instantiate a small scale, 30-node MPR network and compile its trajectory natively:

```python
import jax
import jax.numpy as jnp
from vbi.models.jax import JaxMPRModel, MPRParams, coupling

n_nodes = 30
key = jax.random.PRNGKey(42)
key, w_key = jax.random.split(key)

model = JaxMPRModel(params=MPRParams(), sigma=0.01, dt=0.1)

x0 = jnp.zeros((n_nodes, 2)).at[:, 1].set(-2.0)

weights = jax.random.uniform(w_key, shape=(n_nodes, n_nodes))
weights = weights - jnp.diag(jnp.diag(weights))

def network_coupling(x):
    return coupling.diffusive_coupling(x, weights)

@jax.jit
def run_simulation(x0, keys_array):
    return model.run(x0, keys_array, network_coupling)

step_keys = jax.random.split(key, 1000)
trajectory = run_simulation(x0, step_keys)
```

## What I’m trying to do next

- Make the structure closer to how VBI expects different backends to work
- Add validation and compare results with existing implementations
- Slowly extend this with more features like delays and better batching

## This is still an early version, so I’ll keep improving it based on feedback.

