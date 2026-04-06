"""
Base interfaces for JAX models.
"""
from jax import lax

class JaxNeuralMass:
    """
    Base class for a JAX neural mass.
    It separates the model definition (RHS) from the simulation loop.
    """
    def __init__(self, params, dt=0.1, integrator=None):
        self.params = params
        self.dt = dt
        self.integrator = integrator

    def rhs(self, x, t, inputs):
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
            # Fallback to Euler
            dx = self.rhs(x, 0.0, inputs)
            noise = self.noise(key, x.shape)
            return x + self.dt * dx + noise

    def run(self, x0, keys, coupling_fn):
        """
        Run the simulation using lax.scan.
        """
        def body(x, key):
            inputs = coupling_fn(x)
            x_new = self.step(x, key, inputs)
            return x_new, x_new
        _, traj = lax.scan(body, x0, keys)
        return traj
