"""
BOLD signal models for JAX implementation (Balloon-Windkessel).
"""
from typing import NamedTuple
import jax.numpy as jnp

class BOLDParams(NamedTuple):
    tau_s: float = 0.65
    tau_f: float = 0.41
    tau_o: float = 0.98
    alpha: float = 0.32
    te: float = 0.04
    v0: float = 4.0
    e0: float = 0.4
    epsilon: float = 0.5
    nu_0: float = 40.3
    r_0: float = 25.0

    @property
    def recip_tau_s(self): return 1.0 / self.tau_s
    @property
    def recip_tau_f(self): return 1.0 / self.tau_f
    @property
    def recip_tau_o(self): return 1.0 / self.tau_o
    @property
    def recip_alpha(self): return 1.0 / self.alpha
    @property
    def recip_e0(self): return 1.0 / self.e0

def bold_dfun(sfvq, r_in, p: BOLDParams):
    """
    Computes derivative of Balloon-Windkessel BOLD state.
    sfvq: (4, n_nodes) - the state variables [s, f, v, q]
    r_in: (n_nodes,) - the neural input (firing rate `r`)
    """
    s, f, v, q = sfvq[0], sfvq[1], sfvq[2], sfvq[3]
    
    ds = r_in - p.recip_tau_s * s - p.recip_tau_f * (f - 1)
    df = s
    dv = p.recip_tau_o * (f - v ** p.recip_alpha)
    dq = p.recip_tau_o * (f * (1 - (1 - p.e0) ** (1 / f)) * p.recip_e0
                          - v ** p.recip_alpha * (q / v))
    
    return jnp.stack([ds, df, dv, dq], axis=0)

def bold_euler_step(sfvq, r_in, dt, p: BOLDParams):
    """
    Euler step for BOLD computation.
    """
    dsfvq = bold_dfun(sfvq, r_in, p)
    sfvq_new = sfvq + dt * dsfvq
    
    # Clipping physiological variables to avoid divergence
    s_new = sfvq_new[0]
    f_new = jnp.clip(sfvq_new[1], a_min=1.0)
    v_new = sfvq_new[2]
    q_new = jnp.clip(sfvq_new[3], a_min=0.01)
    
    return jnp.stack([s_new, f_new, v_new, q_new], axis=0)

def compute_bold_signal(sfvq, p: BOLDParams):
    """
    Compute actual BOLD signal observation mapping from (v, q) states.
    """
    v = sfvq[2]
    q = sfvq[3]
    
    k1 = 4.3 * p.nu_0 * p.e0 * p.te
    k2 = p.epsilon * p.r_0 * p.e0 * p.te
    k3 = 1.0 - p.epsilon
    
    bold = p.v0 * (k1 * (1.0 - q) + k2 * (1.0 - q / v) + k3 * (1.0 - v))
    return bold
