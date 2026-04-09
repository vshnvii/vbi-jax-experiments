import time
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from vbi.models.numba.mpr import MPR_sde
from vbi.models.jax.mpr import JaxMPRModel, MPRParams
from vbi.models.jax.coupling import diffusive_coupling

def compute_peak_frequency(signal, dt):
    """
    Compute peak frequency from FFT of the signal.
    """
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=dt)
    fft_vals = np.abs(np.fft.fft(signal))
    # Consider only positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_vals = fft_vals[pos_mask]
    if len(fft_vals) > 0:
        peak_idx = np.argmax(fft_vals)
        return freqs[peak_idx] * 1000  # convert from ms^-1 to Hz
    return 0.0

def run_benchmark():
    n_nodes = 30
    dt = 0.01
    t_end = 2000.0  # 2000 ms to give good oscillation data
    # Create deterministic oscillations: 
    iapp = 3.0
    G = 0.5
    J = 14.5
    eta = -4.6
    tau = 1.0
    delta = 0.7
    noise_amp = 0.0
    
    print(f"--- Initialization ---")
    print(f"Nodes: {n_nodes}, dt: {dt}, t_end: {t_end}")
    
    # 1. Weights
    np.random.seed(42)
    weights = np.random.uniform(0, 1, size=(n_nodes, n_nodes))
    np.fill_diagonal(weights, 0)
    weights /= weights.max()
    
    # Common Initial State
    # Shape of x0 for numba is (2*n_nodes) [r0..rn, v0..vn]
    # Shape of x0 for jax is (n_nodes, 2)
    r_init = np.random.uniform(0, 1.5, size=n_nodes)
    v_init = np.random.uniform(-2, 2, size=n_nodes)
    
    x0_numba = np.concatenate([r_init, v_init])
    x0_jax = jnp.stack([jnp.array(r_init), jnp.array(v_init)], axis=-1)

    # ------------------
    # Numba Execution
    # ------------------
    print("\n[NUMBA] Warmup & Run...")
    numba_model = MPR_sde({
        "weights": weights,
        "dt": dt,
        "t_end": t_end,
        "G": G,
        "J": J,
        "eta": np.full(n_nodes, eta),
        "tau": tau,
        "delta": delta,
        "iapp": iapp,
        "noise_amp": noise_amp,
        "RECORD_BOLD": False,
        "initial_state": x0_numba,
    })
    
    t0 = time.time()
    numba_res = numba_model.run()
    numba_time = time.time() - t0
    
    # Extract timeseries
    rv_d_numba = numba_res["rv_d"]  # (n_steps, 2 * nn)
    # Re-structure NUMBA to (n_steps, nn, 2)
    r_numba = rv_d_numba[:, :n_nodes]
    v_numba = rv_d_numba[:, n_nodes:]
    # The output array naturally aligns dt to length
    
    # ------------------
    # JAX Execution
    # ------------------
    print("\n[JAX] Warmup & Run...")
    jax_params = MPRParams(
        tau=tau,
        I=iapp,
        Delta=delta,
        J=J,
        eta=eta,
        cr=G,   # G couples to r natively in equation
        cv=0.0  # We do not couple V
    )
    
    jax_model = JaxMPRModel(params=jax_params, sigma=0.0, dt=dt)
    
    def coupling_fn(x):
        return diffusive_coupling(x, jnp.array(weights))
        
    n_steps = int(t_end / dt)
    keys = jax.random.split(jax.random.PRNGKey(42), n_steps)
    
    @jax.jit
    def jax_run(x0, keys_arr):
        return jax_model.run(x0, keys_arr, coupling_fn)
        
    # Trigger JIT
    empty_traj = jax_run(x0_jax, keys[:2])
    
    t0 = time.time()
    jax_traj = jax_run(x0_jax, keys)
    
    # Force evaluation since jax returns promises occasionally until retrieved
    jax_r = np.array(jax_traj[:, :, 0])  
    jax_v = np.array(jax_traj[:, :, 1])
    jax_time = time.time() - t0
    
    # ------------------
    # Compatibility & Benchmark Output
    # ------------------
    print(f"\n--- Benchmark Results ---")
    print(f"NUMBA Execution time: {numba_time:.4f} sec")
    print(f"JAX Execution time:   {jax_time:.4f} sec (ignoring JIT compilation)")

    # Numba outputs up to Nt-1 steps
    min_steps = min(r_numba.shape[0], jax_r.shape[0])
    
    # Verify oscillatory behavior on node 0
    signal_numba = r_numba[:min_steps, 0]
    signal_jax = jax_r[:min_steps, 0]
    
    r_var_numba = np.var(signal_numba)
    r_var_jax = np.var(signal_jax)
    
    peak_freq_numba = compute_peak_frequency(signal_numba, dt)
    peak_freq_jax = compute_peak_frequency(signal_jax, dt)

    print(f"\n--- Dynamics Verification ---")
    print(f"NUMBA Node 0: Variance = {r_var_numba:.4e}, Peak Freq = {peak_freq_numba:.2f} Hz")
    print(f"JAX Node 0:   Variance = {r_var_jax:.4e}, Peak Freq = {peak_freq_jax:.2f} Hz")
    
    # Mean Squared Error
    mse_r = np.mean((r_numba[:min_steps] - jax_r[:min_steps])**2)
    mse_v = np.mean((v_numba[:min_steps] - jax_v[:min_steps])**2)
    
    print(f"\n--- Trajectory Consistency ---")
    print(f"MSE (r): {mse_r:.4e}")
    print(f"MSE (v): {mse_v:.4e}")
    
    if mse_r < 1e-4 and mse_v < 1e-4:
        print("✅ SUCCESS: The implementations are numerically highly consistent.")
    else:
        print("❌ FAIL: Significant divergence found between implementations.")
    
    # Oscillation Plot Check
    try:
        t_axis = np.arange(min_steps) * dt
        plt.figure(figsize=(10, 4))
        plt.plot(t_axis, signal_numba, label="Numba", alpha=0.8, linewidth=2)
        plt.plot(t_axis, signal_jax, label="JAX", linestyle='dashed', alpha=0.8, linewidth=2)
        plt.title("Neural Activity Oscillations (Node 0) - I_ext = 3.0")
        plt.xlabel("Time (ms)")
        plt.ylabel("Population Firing Rate (r)")
        plt.legend()
        plt.savefig("oscillation_check.png")
        print("\n=> Saved trajectory plot to 'oscillation_check.png'")
    except Exception as e:
        print("Could not save plot", e)

if __name__ == "__main__":
    run_benchmark()
