"""
Diagnostic run: trace sigma_a, sigma_b, inflate_factor, and Kalman gain
magnitude cycle by cycle, to find exactly where RTPS diverges, instead
of guessing at another blind patch.
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.lorenz96 import integrate_lorenz96, make_initial_condition, generate_observations_96
from src.enkf import rk4_step, build_localization

K, F, dt, N = 40, 8.0, 1e-2, 400  # short run, just enough to find the break
X0_true = make_initial_condition(K, F, perturb=0.0, seed=42)
X0_bg   = make_initial_condition(K, F, perturb=0.05, seed=123)
_, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt)
X_obs, obs_idx = generate_observations_96(X_true, noise_level=0.05, obs_every=2)

N_ens, R_std, loc_radius, alpha, da_interval = 50, 0.05, 8, 0.7, 5

np.random.seed(42)
X_ens = X0_bg[None, :] + np.random.randn(N_ens, K) * 0.01

for step in range(1, N):
    for m in range(N_ens):
        x_new = rk4_step(X_ens[m], dt, K, F)
        if not np.isfinite(x_new).all():
            print(f"step={step}: RK4 forecast produced non-finite value BEFORE any analysis")
            break
        X_ens[m] = x_new

    if step % da_interval == 0:
        x_mean_prior = X_ens.mean(axis=0)
        sigma_b = X_ens.std(axis=0, ddof=1)
        A = X_ens - x_mean_prior

        H = np.zeros((len(obs_idx), K))
        for i, idx in enumerate(obs_idx):
            H[i, idx] = 1.0
        rho = build_localization(K, obs_idx, loc_radius)

        HA = (H @ A.T).T
        Pf = (HA.T @ HA) / (N_ens - 1)
        PfHt = (A.T @ HA) / (N_ens - 1)
        PfHt_loc = PfHt * rho
        R = np.eye(len(obs_idx)) * R_std**2

        Pf_plus_R = Pf + R
        cond = np.linalg.cond(Pf_plus_R)
        K_gain = PfHt_loc @ np.linalg.inv(Pf_plus_R)

        y_obs = X_obs[step, obs_idx]
        Y_pert = y_obs[:, None] + np.random.randn(len(obs_idx), N_ens) * R_std
        innov = Y_pert - H @ X_ens.T
        X_ana = X_ens + (K_gain @ innov).T

        sigma_a = X_ana.std(axis=0, ddof=1)
        max_sigma_b, max_sigma_a = sigma_b.max(), sigma_a.max()
        max_gain = np.abs(K_gain).max()
        max_state = np.abs(X_ana).max()

        if step % 50 == 0 or max_state > 20 or not np.isfinite(max_state):
            print(f"step={step:4d} | cond(Pf+R)={cond:10.2f} | max|K_gain|={max_gain:8.2f} | "
                  f"max(sigma_b)={max_sigma_b:8.3f} | max(sigma_a)={max_sigma_a:8.3f} | "
                  f"max|X_ana|={max_state:10.2f}")

        if not np.isfinite(max_state) or max_state > 1000:
            print(f"\n>>> BLOWUP DETECTED at step={step}, stopping trace here.")
            break

        X_ens = X_ana
