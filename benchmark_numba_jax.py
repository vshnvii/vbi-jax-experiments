import os
import sys
# Force python to load local `vbi` package over installed versions
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import time
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from vbi.models.numba.mpr import MPR_sde
from jax_experiments.mpr import JaxMPRModel, MPRParams

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
    # ---------------------------------------------------------
    # HARDWARE AWARENESS: Dynamic CI vs Local GPU parameters
    # ---------------------------------------------------------
    is_ci_environment = os.environ.get('CI') == 'true'
    current_backend = jax.default_backend()

    if is_ci_environment or current_backend == 'cpu':
        print(f"⚠️  Running on CPU (Backend: {current_backend}). Triggering CI Smoke Test mode.")
        n_nodes = 5         # Reduced nodes for fast testing
        t_end = 100.0       # 1/10th the simulation time
        freq_tol = 20.0     # Looser tolerances for shorter signal FFTs
        var_tol = 30.0
        mse_r_tol = 0.10
        mse_v_tol = 0.30
    else:
        print(f"🚀 Hardware accelerator detected (Backend: {current_backend}). Triggering Full Benchmark.")
        n_nodes = 30        # Your original size
        t_end = 1000.0      # Your original duration
        freq_tol = 10.0     # Your original strict tolerance
        var_tol = 20.0
        mse_r_tol = 0.05
        mse_v_tol = 0.15

    dt = 0.001
    
    # Create deterministic oscillations:
    iapp = 2.2
    G = 0.3
    J = 9.0
    eta = -4.6
    tau = 1.0
    delta = 0.6
    noise_amp = 0.0

    # Decimation factor — applied to BOTH models equally
    rv_decimate = 10

    print(f"\n--- Initialization ---")
    print(f"Nodes: {n_nodes}, dt: {dt}, t_end: {t_end}")

    # 1. Weights
    np.random.seed(42)
    weights = np.random.uniform(0, 1, size=(n_nodes, n_nodes))
    np.fill_diagonal(weights, 0)
    weights /= weights.max()

    # Common Initial State
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
    })

    t0 = time.time()
    numba_res = numba_model.run(x0=x0_numba)
    numba_time = time.time() - t0

    # Extract timeseries
    rv_d_numba = numba_res["rv_d"]  # (n_steps, 2 * nn)

    r_numba = rv_d_numba[::rv_decimate, :n_nodes]
    v_numba = rv_d_numba[::rv_decimate, n_nodes:]

    # ------------------
    # JAX Execution
    # ------------------
    print("\n[JAX] Warmup & Run...")
    jax_params = MPRParams(
        weights=jnp.array(weights),
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
        r = x[:, 0]  # firing rates of all nodes, shape (n_nodes,)
        coupled_r = jnp.dot(jnp.array(weights), r)  # weighted sum from all nodes
        coupled_input = jnp.zeros_like(x)
        coupled_input = coupled_input.at[:, 0].set(coupled_r)
        return coupled_input

    n_steps = int(t_end / dt)
    keys = jax.random.split(jax.random.PRNGKey(42), n_steps)

    @jax.jit
    def jax_run(x0, keys_arr):
        return jax_model.run(x0, keys_arr, coupling_fn)

    # Trigger JIT compilation before timing
    _ = jax_run(x0_jax, keys[:2])

    t0 = time.time()
    jax_traj = jax_run(x0_jax, keys)

    jax_r = np.array(jax_traj[::rv_decimate, :, 0])
    jax_v = np.array(jax_traj[::rv_decimate, :, 1])

    print("JAX min:", np.nanmin(jax_r))
    print("JAX max:", np.nanmax(jax_r))
    print("JAX has NaN:", np.isnan(jax_r).any())
    jax_time = time.time() - t0

    # ------------------
    # Compatibility & Benchmark Output
    # ------------------
    print(f"\n--- Benchmark Results ---")
    print(f"NUMBA Execution time: {numba_time:.4f} sec")
    print(f"JAX Execution time:   {jax_time:.4f} sec (ignoring JIT compilation)")

    min_steps = min(r_numba.shape[0], jax_r.shape[0])

    # Verify oscillatory behavior on node 0
    signal_numba = r_numba[:min_steps, 0]
    signal_jax   = jax_r[:min_steps, 0]

    r_var_numba = np.var(signal_numba)
    r_var_jax   = np.var(signal_jax)

    effective_dt = dt * rv_decimate
    peak_freq_numba = compute_peak_frequency(signal_numba, effective_dt)
    peak_freq_jax   = compute_peak_frequency(signal_jax,   effective_dt)

    print(f"\n--- Dynamics Verification ---")
    print(f"NUMBA Node 0: Variance = {r_var_numba:.4e}, Peak Freq = {peak_freq_numba:.2f} Hz")
    print(f"JAX Node 0:   Variance = {r_var_jax:.4e}, Peak Freq = {peak_freq_jax:.2f} Hz")

    # Frequency and variance agreement
    freq_diff_pct = abs(peak_freq_numba - peak_freq_jax) / (peak_freq_numba + 1e-9) * 100
    var_diff_pct  = abs(r_var_numba - r_var_jax) / (r_var_numba + 1e-9) * 100
    print(f"\nFrequency difference: {freq_diff_pct:.1f}%")
    print(f"Variance difference:  {var_diff_pct:.1f}%")

    # Mean Squared Error
    mse_r = np.mean((r_numba[:min_steps] - jax_r[:min_steps])**2)
    mse_v = np.mean((v_numba[:min_steps] - jax_v[:min_steps])**2)

    print(f"\n--- Trajectory Consistency ---")
    print(f"MSE (r): {mse_r:.4e}")
    print(f"MSE (v): {mse_v:.4e}")

    # Dynamic Success Check
    if freq_diff_pct < freq_tol and var_diff_pct < var_tol:
        print(f"SUCCESS: Both models show consistent oscillatory dynamics within {freq_tol}% tolerance.")
    elif mse_r < mse_r_tol and mse_v < mse_v_tol:
        print("PARTIAL: MSE is acceptable but dynamics differ slightly — check the plot.")
    else:
        print(f"FAIL: Significant divergence found between implementations.")
        sys.exit(1) 

    # Oscillation Plot Check
    try:
        t_axis = np.arange(min_steps) * effective_dt
        plt.figure(figsize=(10, 4))
        plt.plot(t_axis, signal_numba, label="Numba", alpha=0.8, linewidth=2)
        plt.plot(t_axis, signal_jax, label="JAX", linestyle='dashed', alpha=0.8, linewidth=2)
        plt.title("Neural Activity Oscillations (Node 0) - I_ext = 2.2")
        plt.xlabel("Time (ms)")
        plt.ylabel("Population Firing Rate (r)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("oscillation_check.png")
        print("\n=> Saved trajectory plot to 'oscillation_check.png'")
    except Exception as e:
        print("Could not save plot:", e)

if __name__ == "__main__":
    run_benchmark()