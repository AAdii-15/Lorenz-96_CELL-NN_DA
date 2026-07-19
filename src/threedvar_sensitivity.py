"""
3D-Var covariance sensitivity study -- corrected schedule
=============================================================
Tests whether the 3.7% Lorenz-96 Cell-NN advantage survives a
properly tuned 3D-Var baseline. Grid search over (pb_std, r_std),
same spirit as Cell-NN's (alpha_A, w_b) search (Figure 2) and the
EnKF inflation/localization sweeps (Table 3). Uses the intermittency
schedule (DA-on 0-2000, off 2000-3500, on 3500-5000), matching every
other Table 3 comparison -- confirmed via sanity check to reproduce
the baseline (pb_std=0.5, r_std=0.1) R-score of 1.411 (got 1.4106).
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.lorenz96 import integrate_lorenz96, make_initial_condition, generate_observations_96
from src.lorenz96_da import run_3dvar_lorenz96
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("3D-Var covariance sensitivity study (intermittency schedule)")
print("=" * 70)

K, F, dt, N = 40, 8.0, 1e-2, 5000
X0_true = make_initial_condition(K, F, perturb=0.0, seed=42)
X0_bg   = make_initial_condition(K, F, perturb=0.05, seed=123)
_, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt)
X_obs, obs_idx = generate_observations_96(X_true, noise_level=0.05, obs_every=2)

da_interval = 25
da_on_periods = [(0, 2000), (3500, 5000)]
da_on_mask = np.zeros(N, dtype=bool)
for s, e in da_on_periods:
    da_on_mask[s:e] = True

PB_VALS = [0.1, 0.3, 0.5, 0.7, 1.0]
R_VALS  = [0.05, 0.1, 0.2, 0.3, 0.5]
BASELINE = (0.5, 0.1)

print(f"\n{'pb_std':>8}  {'r_std':>8}  {'R-score':>10}")
print("-" * 32)

results = {}
for pb in PB_VALS:
    for r in R_VALS:
        w_sim, _ = run_3dvar_lorenz96(
            X_true, X_obs, obs_idx, X0_bg, dt,
            da_interval=da_interval, da_on_periods=da_on_periods,
            pb_std=pb, r_std=r, K=K, F=F)
        R_score = float(np.mean((X_true[da_on_mask] - w_sim[da_on_mask])**2))
        results[(pb, r)] = R_score
        mark = '  <-- baseline (Table 3: 1.411)' if (pb, r) == BASELINE else ''
        print(f"{pb:>8.2f}  {r:>8.2f}  {R_score:>10.4f}{mark}")

print(f"\n{'='*70}")
print("FULL GRID (rows=pb_std, cols=r_std)")
print(f"{'':>10}", end="")
for r in R_VALS:
    print(f"  r_std={r:.2f}", end="")
print()
for pb in PB_VALS:
    print(f"pb_std={pb:.2f}", end="  ")
    for r in R_VALS:
        print(f"  {results[(pb,r)]:>9.4f}", end="")
    print()
print(f"{'='*70}")

best = min(results, key=results.get)
print(f"\nBaseline (pb_std=0.5, r_std=0.1): R-score={results[BASELINE]:.4f}")
print(f"Best in grid: pb_std={best[0]}, r_std={best[1]}, R-score={results[best]:.4f}")
print(f"Cell-NN R-score (irregular, Table 3): 1.245")
improvement_needed = results[BASELINE] - 1.245
print(f"Best-in-grid vs Cell-NN: {'3D-Var still worse' if results[best] > 1.245 else '3D-Var now BEATS Cell-NN'} "
      f"(best={results[best]:.4f} vs Cell-NN=1.245)")

np.savez('results/threedvar_sensitivity.npz',
         pb_vals=np.array(PB_VALS), r_vals=np.array(R_VALS),
         rmse_grid=np.array([[results[(pb,r)] for r in R_VALS] for pb in PB_VALS]))
print("\nSaved -> results/threedvar_sensitivity.npz")
print("Done.")
