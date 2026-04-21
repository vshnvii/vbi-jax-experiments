import time
import jax
import jax.numpy as jnp
from vbi.models.jax.mpr import JaxMPRModel, MPRParams
from vbi.models.jax.coupling import diffusive_coupling

def verify_simulation():
    # 10 to 50 nodes
    n_nodes = 30
    print(f"Initializing MPR JAX model for {n_nodes} nodes.")
    
    # Setup model
    W = jax.random.uniform(jax.random.PRNGKey(42), shape=(n_nodes, n_nodes))
    W = W - jnp.diag(jnp.diag(W)) # zero diagonal
    W = W / jnp.max(W) # Normalize

    params = MPRParams(weights=W)
    model = JaxMPRModel(params=params, sigma=0.01, dt=0.1)
    
    # Initial state (r, V). Note: MPR State -> (r, V)
    # r = 0.0, V = -2.0 is default from vbjax
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    
    x0 = jnp.zeros((n_nodes, 2))
    x0 = x0.at[:, 0].set(0.0)
    x0 = x0.at[:, 1].set(-2.0)
    
    # Set up coupling randomly for nodes (already generated above for parameters)
    
    def coupling_fn(x):
        return diffusive_coupling(x, W)
    
    # Simulation length
    n_steps = 1000
    keys = jax.random.split(key, n_steps)
    
    print("Compiling and Running Simulation using lax.scan ...")
    start = time.time()
    # To properly JIT the entire simulation 
    @jax.jit
    def run_sim(x0, keys):
        return model.run(x0, keys, coupling_fn)
        
    traj = run_sim(x0, keys)
    end = time.time()
    
    print(f"Simulation completed in {end - start:.4f} seconds.")
    print(f"Trajectory shape: {traj.shape}")  # should be (n_steps, n_nodes, 2)
    print("Simulation execution successful!")

if __name__ == "__main__":
    verify_simulation()
