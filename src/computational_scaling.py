"""
Computational scaling: wall-clock time vs. state dimension K
==================================================================
Lighter, separate run from computational_cost.py -- N=1000 steps,
1 repeat, K in {10,20,40,80,160} -- built specifically to show the
SCALING TREND with K, not to reproduce Table 5's exact absolute
timings (which use the full N=5000, 3 repeats, K=40 only). Confirms
Table 5's stated asymptotic complexity: Cell-NN O(K), 3D-Var O(K^2),
EnKF O(N_ens*K) + forecast cost.
"""
import numpy as np
import time, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.lorenz96 import (integrate_lorenz96, make_initial_condition,
                           generate_observations_96)
from src.lorenz96_da import run_3dvar_lorenz96, run_cellnn_lorenz96
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("Computational Scaling: wall-clock time vs. K")
print("=" * 65)

K_VALS = [10, 20, 40, 80, 160]
F, dt, N = 8.0, 1e-2, 1000
da_interval = 25

results_3dvar, results_cellnn, results_enkf = {}, {}, {}

for K in K_VALS:
    print(f"\n--- K={K} ---")
    X0_true = make_initial_condition(K, F, perturb=0.0, seed=42)
    X0_bg   = make_initial_condition(K, F, perturb=0.05, seed=123)
    _, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt, K=K, F=F)
    X_obs, obs_idx = generate_observations_96(X_true, noise_level=0.05, obs_every=2)
    da_on_periods = [(0, N)]  # continuous DA, timing only, not R-score

    t0 = time.perf_counter()
    run_3dvar_lorenz96(X_true, X_obs, obs_idx, X0_bg, dt,
                        da_interval=da_interval, da_on_periods=da_on_periods,
                        K=K, F=F)
    t_3dvar = time.perf_counter() - t0
    results_3dvar[K] = t_3dvar
    print(f"  3D-Var:  {t_3dvar:.3f}s")

    t0 = time.perf_counter()
    run_cellnn_lorenz96(X_true, X_obs, obs_idx, X0_bg, dt,
                         da_interval=da_interval, da_on_periods=da_on_periods,
                         alpha_A=1.0, wb=1.0, K=K, F=F)
    t_cellnn = time.perf_counter() - t0
    results_cellnn[K] = t_cellnn
    print(f"  Cell-NN: {t_cellnn:.3f}s")

    try:
        from src.lorenz96_da import run_enkf_lorenz96_silent
        t0 = time.perf_counter()
        run_enkf_lorenz96_silent(X_true, X_obs, obs_idx, X0_bg, dt,
                                  da_interval=da_interval, da_on_periods=da_on_periods,
                                  K=K, F=F, N_ens=50)
        t_enkf = time.perf_counter() - t0
    except ImportError:
        from src.enkf import run_enkf_lorenz96
        t0 = time.perf_counter()
        run_enkf_lorenz96(X_true, X_obs, obs_idx, X0_bg, dt,
                           da_interval=da_interval, da_on_periods=da_on_periods,
                           K=K, F=F, N_ens=50)
        t_enkf = time.perf_counter() - t0
    results_enkf[K] = t_enkf
    print(f"  EnKF:    {t_enkf:.3f}s")

print(f"\n{'='*65}")
print("SUMMARY -- wall time (s) vs K")
print(f"{'='*65}")
print(f"{'K':>6}{'3D-Var':>12}{'Cell-NN':>12}{'EnKF':>12}")
for K in K_VALS:
    print(f"{K:>6}{results_3dvar[K]:>12.3f}{results_cellnn[K]:>12.3f}{results_enkf[K]:>12.3f}")

np.savez('results/computational_scaling.npz',
         K_vals=np.array(K_VALS),
         t_3dvar=np.array([results_3dvar[K] for K in K_VALS]),
         t_cellnn=np.array([results_cellnn[K] for K in K_VALS]),
         t_enkf=np.array([results_enkf[K] for K in K_VALS]))
print("\nSaved -> results/computational_scaling.npz")
print("Done.")
