"""
Diagnostic run #2: actually calls the real enkf_analysis_rtps(), traces
sigma_b, sigma_a, and the resulting inflate_factor every cycle, over
enough cycles to catch compounding growth (not just a single-step spike).
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.lorenz96 import integrate_lorenz96, make_initial_condition, generate_observations_96
from src.enkf import rk4_step
from src.enkf_rtps import enkf_analysis_rtps

K, F, dt, N = 40, 8.0, 1e-2, 1200  # ~240 analysis cycles
X0_true = make_initial_condition(K, F, perturb=0.0, seed=42)
X0_bg   = make_initial_condition(K, F, perturb=0.05, seed=123)
_, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt)
X_obs, obs_idx = generate_observations_96(X_true, noise_level=0.05, obs_every=2)

N_ens, R_std, loc_radius, alpha, da_interval = 50, 0.05, 8, 0.7, 5

np.random.seed(42)
X_ens = X0_bg[None, :] + np.random.randn(N_ens, K) * 0.01
cycle = 0

for step in range(1, N):
    for m in range(N_ens):
        x_new = rk4_step(X_ens[m], dt, K, F)
        if not np.isfinite(x_new).all():
            print(f"step={step}: forecast produced non-finite value")
            x_new = X_ens.mean(axis=0) + np.random.randn(K) * 0.01
        X_ens[m] = x_new

    if step % da_interval == 0:
        cycle += 1
        sigma_b_pre = X_ens.std(axis=0, ddof=1)
        y_obs = X_obs[step, obs_idx]
        X_ana_relaxed, x_mean = enkf_analysis_rtps(
            X_ens, y_obs, obs_idx, R_std=R_std, loc_radius=loc_radius, alpha=alpha)

        sigma_after = X_ana_relaxed.std(axis=0, ddof=1)
        max_state = np.abs(X_ana_relaxed).max()

        if cycle % 20 == 0 or max_state > 20 or not np.isfinite(max_state):
            print(f"cycle={cycle:4d} step={step:4d} | max(sigma_b_pre)={sigma_b_pre.max():8.3f} | "
                  f"max(sigma_after_RTPS)={sigma_after.max():8.3f} | max|X_ana|={max_state:10.3f}")

        if not np.isfinite(max_state) or max_state > 1000:
            print(f"\n>>> BLOWUP at cycle={cycle}, step={step}. Stopping trace.")
            break

        X_ens = X_ana_relaxed
