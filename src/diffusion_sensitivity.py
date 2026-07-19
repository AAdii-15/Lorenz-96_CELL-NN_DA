"""
Diffusion Hyperparameter Sensitivity
=====================================
Addresses Flaw 9: diffusion is the sole skill driver but κ=0.3, 
n_diff=30 have never been ablated. 
Tests κ ∈ {0.1, 0.2, 0.3, 0.4} x n_diff ∈ {10, 20, 30, 50}.
Same 20% test mask (seed=42) as all prior runs.
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath('.')))
from src.da_cellnn import cellnn_da_step
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("Experiment A: Diffusion Hyperparameter Sensitivity")
print("=" * 65)

chl_norm = np.load('data/modis_chl/chl_norm.npy')
N, LAT, LON = chl_norm.shape

# Reproduce identical test mask
np.random.seed(42)
test_mask = np.zeros((N, LAT, LON), dtype=bool)
eval_t = []
for t in range(1, N):
    field_curr = chl_norm[t]
    valid_mask = ~np.isnan(field_curr)
    if valid_mask.sum() < 500:
        continue
    valid_idx = np.where(valid_mask)
    n_valid = len(valid_idx[0])
    sel = np.random.choice(n_valid, int(0.2 * n_valid), replace=False)
    test_mask[t, valid_idx[0][sel], valid_idx[1][sel]] = True
    eval_t.append(t)

print(f"Test timesteps: {len(eval_t)}")

KAPPA_VALS  = [0.1, 0.2, 0.3, 0.4]
NDIFF_VALS  = [10, 20, 30, 50]

def diffusion_only(mu_b_2d, y_obs_2d, obs_mask_2d, kappa, n_diff):
    """Pure diffusion — no Cell-NN (confirmed as sole skill driver)."""
    mu_a = mu_b_2d.copy()
    mu_a[obs_mask_2d] = y_obs_2d[obs_mask_2d]
    for _ in range(n_diff):
        p = np.pad(mu_a, 1, mode='edge')
        L = p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*mu_a
        u = np.zeros_like(mu_a)
        u[~obs_mask_2d] = kappa * L[~obs_mask_2d]
        mu_a = mu_a + 0.1 * u
    return mu_a

results = {}
print(f"\n{'kappa':>8}  {'n_diff':>8}  {'RMSE':>10}")
print("-" * 30)

for kappa in KAPPA_VALS:
    for n_diff in NDIFF_VALS:
        rmse_list = []
        for t in eval_t:
            fc = chl_norm[t]; fp = chl_norm[t-1]
            valid_mask = ~np.isnan(fc)
            tm = test_mask[t]
            mu_b = fp.copy(); mu_b[np.isnan(mu_b)] = 0.0
            obs_mask = valid_mask & (~tm)
            y_obs = np.where(obs_mask, fc, 0.0)
            mu_a = diffusion_only(mu_b, y_obs, obs_mask, kappa, n_diff)
            rows, cols = np.where(tm)
            true_v = fc[rows, cols]
            pred_v = mu_a[rows, cols]
            rmse_list.append(np.sqrt(np.mean((true_v - pred_v)**2)))
        rmse_mean = np.mean(rmse_list)
        results[(kappa, n_diff)] = rmse_mean
        mark = ' <-- baseline' if kappa == 0.3 and n_diff == 30 else ''
        print(f"  {kappa:>6.1f}  {n_diff:>8d}  {rmse_mean:>10.4f}{mark}")

print(f"\n{'='*65}")
print("FULL GRID (rows=kappa, cols=n_diff)")
print(f"{'':>10}", end="")
for nd in NDIFF_VALS:
    print(f"  n_diff={nd:2d}", end="")
print()
for kappa in KAPPA_VALS:
    print(f"kappa={kappa:.1f}", end="  ")
    for n_diff in NDIFF_VALS:
        print(f"  {results[(kappa,n_diff)]:>10.4f}", end="")
    print()
print(f"{'='*65}")

best = min(results, key=results.get)
print(f"\nBest config: kappa={best[0]}, n_diff={best[1]}, RMSE={results[best]:.4f}")
print(f"Baseline config (kappa=0.3, n_diff=30): RMSE={results[(0.3,30)]:.4f}")
print(f"Range: {min(results.values()):.4f} to {max(results.values()):.4f}")

np.savez('results/diffusion_sensitivity.npz',
         kappa_vals=np.array(KAPPA_VALS),
         ndiff_vals=np.array(NDIFF_VALS),
         rmse_grid=np.array([[results[(k,n)] for n in NDIFF_VALS] 
                              for k in KAPPA_VALS]))
print("\nSaved -> results/diffusion_sensitivity.npz")
print("\nDone.")
