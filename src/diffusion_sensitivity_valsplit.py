"""
Diffusion Hyperparameter Sensitivity -- VALIDATION SPLIT (leakage fix)
========================================================================
Identical grid and identical diffusion_only() logic to
src/diffusion_sensitivity.py. The only change: scores each (kappa, n_diff)
against val_mask (10% of pixels) instead of the full 20% held-out mask,
so the winning config is chosen without ever touching the test set.
"""
import numpy as np

chl_norm = np.load('data/modis_chl/chl_norm.npy')
N, LAT, LON = chl_norm.shape

split = np.load('results/holdout_split_masks.npz')
held_out_20 = split['held_out_20']
val_mask    = split['val_mask']
eval_t      = split['eval_t']

print(f"Loaded split masks. Timesteps: {len(eval_t)}")
print(f"Validation pixels: {val_mask.sum()}")

KAPPA_VALS  = [0.1, 0.2, 0.3, 0.4]
NDIFF_VALS  = [10, 20, 30, 50]

def diffusion_only(mu_b_2d, y_obs_2d, obs_mask_2d, kappa, n_diff):
    """Identical to src/diffusion_sensitivity.py -- unchanged."""
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
print(f"\n{'kappa':>8}  {'n_diff':>8}  {'RMSE (val)':>10}")
print("-" * 32)

for kappa in KAPPA_VALS:
    for n_diff in NDIFF_VALS:
        rmse_list = []
        for t in eval_t:
            fc = chl_norm[t]; fp = chl_norm[t-1]
            valid_mask = ~np.isnan(fc)
            # obs_mask excludes the FULL 20% (val+test), same as original
            obs_mask = valid_mask & (~held_out_20[t])
            y_obs = np.where(obs_mask, fc, 0.0)
            mu_b = fp.copy(); mu_b[np.isnan(mu_b)] = 0.0
            mu_a = diffusion_only(mu_b, y_obs, obs_mask, kappa, n_diff)
            # score ONLY on validation pixels
            rows, cols = np.where(val_mask[t])
            true_v = fc[rows, cols]
            pred_v = mu_a[rows, cols]
            rmse_list.append(np.sqrt(np.mean((true_v - pred_v)**2)))
        rmse_mean = np.mean(rmse_list)
        results[(kappa, n_diff)] = rmse_mean
        mark = ' <-- old baseline' if kappa == 0.3 and n_diff == 30 else ''
        mark2 = ' <-- OLD winner' if kappa == 0.4 and n_diff == 50 else ''
        print(f"  {kappa:>6.1f}  {n_diff:>8d}  {rmse_mean:>10.4f}{mark}{mark2}")

print(f"\n{'='*65}")
print("FULL GRID -- VALIDATION SET (rows=kappa, cols=n_diff)")
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
print(f"\nBest config on VALIDATION set: kappa={best[0]}, n_diff={best[1]}, RMSE={results[best]:.4f}")

np.savez('results/diffusion_sensitivity_valsplit.npz',
         kappa_vals=np.array(KAPPA_VALS),
         ndiff_vals=np.array(NDIFF_VALS),
         rmse_grid=np.array([[results[(k,n)] for n in NDIFF_VALS]
                              for k in KAPPA_VALS]),
         best_kappa=best[0], best_ndiff=best[1])
print("\nSaved -> results/diffusion_sensitivity_valsplit.npz")
print("Done.")
