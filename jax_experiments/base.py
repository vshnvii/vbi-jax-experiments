"""
Base interfaces for JAX models.
"""
from jax import lax
import jax.numpy as jnp
from .bold import bold_euler_step, compute_bold_signal

class JaxNeuralMass:
    """
    Base class for a JAX neural mass.
    It separates the model definition (RHS) from the simulation loop.
    Employs strict memory constraints handling standalone BOLD.
    """
    def __init__(self, params, dt=0.1, integrator=None):
        self.params = params
        self.dt = dt
        self.integrator = integrator

    def rhs(self, x, t, params, inputs):
        """dx/dt = f(x, t, params, inputs)"""
        raise NotImplementedError

    def noise(self, key, shape):
        """Stochastic term"""
        raise NotImplementedError

    def step(self, x, key, inputs):
        """Integration step"""
        if self.integrator is not None:
            return self.integrator(x, 0.0, self.dt, self.rhs, self.noise, self.params, inputs, key)
        else:
            dx = self.rhs(x, 0.0, self.params, inputs)
            noise = self.noise(key, x.shape)
            return x + self.dt * dx + noise

    def run(self, x0, keys, coupling_fn, bold_params=None, bold_dt=None, record_rv=True):
        """
        Run the simulation using lax.scan.
        If bold_params is supplied but record_rv is False, only BOLD is retained,
        meaning only the last time steps of `r` are cached in local execution memory.
        """
        if bold_params is not None and bold_dt is None:
            bold_dt = self.dt / 1000.0  # Assumes dt in ms, bold_dt typically in s if tau in Hz
            
        def body(carry, key):
            x, bold_state = carry
            inputs = coupling_fn(x)
            
            x_new = self.step(x, key, inputs)
            
            # BOLD mapping uses firing rate `r` usually sitting at index 0 
            if bold_params is not None:
                r_in = x[..., 0] 
                bold_state_new = bold_euler_step(bold_state, r_in, bold_dt, bold_params)
                bold_out = compute_bold_signal(bold_state_new, bold_params)
            else:
                bold_state_new = None
                bold_out = None
                
            # Condition emission to optimize sequence retention
            if record_rv and (bold_params is not None):
                out = (x_new, bold_out)
            elif record_rv:
                out = x_new
            elif bold_params is not None:
                out = bold_out
            else:
                out = jnp.zeros(0)

            # Carry state: Contains only immediate X history
            return (x_new, bold_state_new), out

        # BOLD default resting states (s=0, f=1, v=1, q=1)
        if bold_params is not None:
            n_nodes = x0.shape[0]
            b_s = jnp.zeros((n_nodes,))
            b_f = jnp.ones((n_nodes,))
            b_v = jnp.ones((n_nodes,))
            b_q = jnp.ones((n_nodes,))
            init_bold = jnp.stack([b_s, b_f, b_v, b_q], axis=0)
        else:
            init_bold = None

        final_carry, traj = lax.scan(body, (x0, init_bold), keys)
        return traj
