"""
Cell-NN reconstruction pipeline -- TEST SPLIT (leakage fix)
==============================================================
Identical cellnn_da_2d() logic to src/cellnn_chl_da_proper.py.
kappa=0.4, n_diff=50 -- unchanged winner from the validation-split
grid search (src/diffusion_sensitivity_valsplit.py).
Scores ONLY on test_mask (10%), which was never touched during
hyperparameter selection. This produces the corrected Background
and Diffusion-pipeline rows of Table 6.
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.da_cellnn import cellnn_da_step

chl_norm = np.load('data/modis_chl/chl_norm.npy')

split = np.load('results/holdout_split_masks.npz')
held_out_20 = split['held_out_20']
test_mask   = split['test_mask']
eval_t      = split['eval_t']

print(f"Test timesteps: {len(eval_t)}")
print(f"Test pixels:    {test_mask.sum()}")

def cellnn_da_2d(mu_b_2d, y_obs_2d, obs_mask_2d,
                  alpha_A=1.0, wb=1.0, kappa=0.4, n_diff=50):
    """Identical to src/cellnn_chl_da_proper.py -- unchanged."""
    lat_n, lon_n = mu_b_2d.shape
    mu_b_flat  = mu_b_2d.flatten()
    y_obs_flat = y_obs_2d.flatten()
    obs_flat   = obs_mask_2d.flatten()
    mu_b_obs = mu_b_flat[obs_flat]
    y_obs    = y_obs_flat[obs_flat]
    mu_a_obs, n_iter = cellnn_da_step(
        mu_b_obs, y_obs, alpha_A=alpha_A, wb=wb,
        d_tau=5e-2, tau_max=1e-1, eps_conv=1e-6, r_max=50)
    mu_a_flat = mu_b_flat.copy()
    mu_a_flat[obs_flat] = mu_a_obs
    mu_a_2d = mu_a_flat.reshape(lat_n, lon_n)
    for _ in range(n_diff):
        mu_pad = np.pad(mu_a_2d, 1, mode='edge')
        laplacian = (mu_pad[:-2,1:-1] + mu_pad[2:,1:-1] +
                     mu_pad[1:-1,:-2] + mu_pad[1:-1,2:] - 4.0*mu_a_2d)
        update = np.zeros_like(mu_a_2d)
        update[~obs_mask_2d] = kappa * laplacian[~obs_mask_2d]
        mu_a_2d = mu_a_2d + 0.1 * update
    return mu_a_2d, n_iter

rmse_cellnn, rmse_bg = [], []

for t in eval_t:
    field_curr = chl_norm[t]
    field_prev = chl_norm[t-1]
    valid_mask = ~np.isnan(field_curr)
    # obs_mask excludes the FULL original 20% (val+test) -- unchanged
    # from the original pipeline; the model still never sees these.
    obs_mask = valid_mask & (~held_out_20[t])
    y_obs = np.where(obs_mask, field_curr, 0.0)
    mu_b = field_prev.copy(); mu_b[np.isnan(mu_b)] = 0.0

    mu_a, n_iter = cellnn_da_2d(mu_b, y_obs, obs_mask,
                                 alpha_A=1.0, wb=1.0, kappa=0.4, n_diff=50)

    # score ONLY on the held-out TEST half
    rows, cols = np.where(test_mask[t])
    true_v = field_curr[rows, cols]
    an_v   = mu_a[rows, cols]
    bg_v   = mu_b[rows, cols]
    rmse_cellnn.append(np.sqrt(np.mean((true_v-an_v)**2)))
    rmse_bg.append(np.sqrt(np.mean((true_v-bg_v)**2)))

rmse_cellnn = np.array(rmse_cellnn)
rmse_bg     = np.array(rmse_bg)
improvement = (rmse_bg.mean()-rmse_cellnn.mean())/rmse_bg.mean()*100

print(f"\n{'='*50}")
print(f"Background RMSE (test-only): {rmse_bg.mean():.4f}")
print(f"Cell-NN pipeline RMSE (test-only): {rmse_cellnn.mean():.4f}")
print(f"Improvement: {improvement:.1f}%")
print(f"{'='*50}")
print(f"\nFor comparison, ORIGINAL (leaky) numbers were:")
print(f"Background RMSE: 0.755, Diffusion pipeline RMSE: 0.134, Improvement: 82.3%")

np.savez('results/chl_pipeline_testsplit.npz',
         rmse_cellnn=rmse_cellnn, rmse_bg=rmse_bg)
print("\nSaved -> results/chl_pipeline_testsplit.npz")
print("Done.")
