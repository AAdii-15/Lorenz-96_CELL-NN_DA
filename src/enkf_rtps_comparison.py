"""
RTPS adaptive-inflation EnKF vs. fixed-inflation EnKF, continuous vs
irregular observations. Same protocol as src/enkf_sensitivity.py's
baseline row (Table 3). Uses the stabilized enkf_rtps.py (member-
deviation cap + forecast-level state ceiling, both grounded in this
paper's own O(1-10) Lorenz-96 state magnitude).
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.lorenz96 import (integrate_lorenz96, make_initial_condition,
                           generate_observations_96)
from src.enkf_rtps import run_enkf_rtps_lorenz96
import warnings
warnings.filterwarnings('ignore')

print("=" * 75)
print("RTPS Adaptive-Inflation EnKF: continuous vs irregular observations")
print("=" * 75)

K, F, dt, N = 40, 8.0, 1e-2, 5000
X0_true = make_initial_condition(K, F, perturb=0.0, seed=42)
X0_bg   = make_initial_condition(K, F, perturb=0.05, seed=123)
_, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt)
X_obs, obs_idx = generate_observations_96(X_true, noise_level=0.05, obs_every=2)

da_on_periods = [(0, 2000), (3500, 5000)]
da_on_mask = np.zeros(N, dtype=bool)
for s, e in da_on_periods:
    da_on_mask[s:e] = True

ALPHA_VALS = [0.1, 0.2, 0.3]

print(f"\n{'alpha':>8}  {'Cont. R-score':>14}  {'Interm. R-score':>16}  {'Ratio':>8}  {'Finite?':>8}")
print("-" * 68)

results = {}
for alpha in ALPHA_VALS:
    X_cont = run_enkf_rtps_lorenz96(
        X_true, X_obs, obs_idx, X0_bg, dt,
        da_interval=5, da_on_periods=[(0, N)],
        K=K, F=F, N_ens=50, R_std=0.05, loc_radius=8, alpha=alpha)
    X_interm = run_enkf_rtps_lorenz96(
        X_true, X_obs, obs_idx, X0_bg, dt,
        da_interval=5, da_on_periods=da_on_periods,
        K=K, F=F, N_ens=50, R_std=0.05, loc_radius=8, alpha=alpha)

    finite = np.isfinite(X_cont).all() and np.isfinite(X_interm).all()
    R_cont = float(np.mean((X_true[da_on_mask] - X_cont[da_on_mask])**2))
    R_interm = float(np.mean((X_true[da_on_mask] - X_interm[da_on_mask])**2))
    ratio = R_interm / max(R_cont, 1e-10)
    results[alpha] = (R_cont, R_interm, ratio, finite)
    verdict = 'DIVERGES' if R_interm > 5 else 'stable'
    print(f"{alpha:>8.1f}  {R_cont:>14.4f}  {R_interm:>16.4f}  {ratio:>7.1f}x  {str(finite):>8}  {verdict}")

print(f"\n{'='*75}")
print("Compare to fixed-inflation EnKF baseline (Table 3):")
print("  Cont R-score=0.005, Interm R-score=23.965, Ratio=4699x, DIVERGES")
print(f"{'='*75}")

np.savez('results/enkf_rtps_comparison.npz',
         alpha_vals=np.array(ALPHA_VALS),
         results=np.array([[a, rc, ri, rt] for a, (rc, ri, rt, fin) in results.items()]))
print("\nSaved -> results/enkf_rtps_comparison.npz")
print("Done.")
