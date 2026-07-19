"""
Cell-NN climatology-blended fallback -- TEST SPLIT + kappa correction
==========================================================================
Two fixes from the original cellnn_hybrid_fallback.py:
  1. kappa=0.4, n_diff=50 (was stale kappa=0.3, n_diff=30 -- same bug
     as stratified_comparison.py).
  2. beta=0.4, kept equal to kappa per the file's own stated design
     rule ("beta = kappa ... chosen for symmetry") -- not a new value,
     just honoring that rule under the corrected kappa.
Comparison baselines (Cell-NN-orig, Monte-Carlo) now loaded from
results/stratified_comparison_testsplit.npz (the corrected run), not
the original leaky/wrong-kappa results/stratified_comparison.npz.
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.da_cellnn import cellnn_da_step
import warnings
warnings.filterwarnings('ignore')

chl_norm = np.load('data/modis_chl/chl_norm.npy')
N, LAT, LON = chl_norm.shape

split = np.load('results/holdout_split_masks.npz')
held_out_20 = split['held_out_20']
test_mask   = split['test_mask']
eval_t      = list(split['eval_t'])

obs_mask_all = (~np.isnan(chl_norm)) & (~held_out_20)
print(f"Test pixels: {int(test_mask.sum()):,} across {len(eval_t)} timesteps")

clim_mean_2d = np.nanmean(np.where(obs_mask_all, chl_norm, np.nan), axis=0)
clim_valid = ~np.isnan(clim_mean_2d)
print(f"Climatology defined for {int(clim_valid.sum())}/{LAT*LON} pixels")

KAPPA, N_DIFF = 0.4, 50
BETA = KAPPA   # kept equal to kappa per original design rule

def cellnn_da_2d_hybrid(mu_b_2d, y_obs_2d, obs_mask_2d, alpha_A=1.0, wb=1.0,
                         kappa=KAPPA, beta=BETA, n_diff=N_DIFF):
    lat_n, lon_n = mu_b_2d.shape
    mu_b_flat = mu_b_2d.flatten()
    y_obs_flat = y_obs_2d.flatten()
    obs_flat = obs_mask_2d.flatten()
    mu_b_obs = mu_b_flat[obs_flat]
    y_obs = y_obs_flat[obs_flat]
    mu_a_obs, n_iter = cellnn_da_step(mu_b_obs, y_obs, alpha_A=alpha_A, wb=wb,
                                       d_tau=5e-2, tau_max=1e-1,
                                       eps_conv=1e-6, r_max=50)
    mu_a_flat = mu_b_flat.copy()
    mu_a_flat[obs_flat] = mu_a_obs
    mu_a_2d = mu_a_flat.reshape(lat_n, lon_n)

    for _ in range(n_diff):
        mu_pad = np.pad(mu_a_2d, 1, mode='edge')
        laplacian = (mu_pad[:-2, 1:-1] + mu_pad[2:, 1:-1] +
                     mu_pad[1:-1, :-2] + mu_pad[1:-1, 2:] - 4.0 * mu_a_2d)
        clim_pull = np.zeros_like(mu_a_2d)
        clim_pull[clim_valid] = clim_mean_2d[clim_valid] - mu_a_2d[clim_valid]

        update = np.zeros_like(mu_a_2d)
        gap = ~obs_mask_2d
        update[gap] = kappa * laplacian[gap] + beta * clim_pull[gap]
        mu_a_2d = mu_a_2d + 0.1 * update

    return mu_a_2d, n_iter

print(f"\nRunning Cell-NN (diffusion + climatology fallback, kappa=beta={KAPPA}, n_diff={N_DIFF})...")

cellnn_hybrid_true, cellnn_hybrid_pred = [], []
rmse_per_t = []

for t in eval_t:
    field_curr = chl_norm[t]
    field_prev = chl_norm[t - 1]
    tm = test_mask[t]
    valid_mask = ~np.isnan(field_curr)

    mu_b = field_prev.copy(); mu_b[np.isnan(mu_b)] = 0.0
    obs_mask = valid_mask & (~held_out_20[t])
    y_obs = np.where(obs_mask, field_curr, 0.0)

    mu_a, n_iter = cellnn_da_2d_hybrid(mu_b, y_obs, obs_mask)

    rows, cols = np.where(tm)
    if len(rows) == 0:
        continue
    true_v = field_curr[rows, cols]
    pred_v = mu_a[rows, cols]
    cellnn_hybrid_true.append(true_v)
    cellnn_hybrid_pred.append(pred_v)
    rmse_per_t.append(np.sqrt(np.mean((true_v - pred_v) ** 2)))

cellnn_hybrid_true = np.concatenate(cellnn_hybrid_true)
cellnn_hybrid_pred = np.concatenate(cellnn_hybrid_pred)
rmse_per_t = np.array(rmse_per_t)

print(f"Done: {len(cellnn_hybrid_true)} test instances")
print(f"Mean RMSE (per-timestep avg): {rmse_per_t.mean():.4f}")
r_hybrid_all = np.sqrt(np.mean((cellnn_hybrid_true - cellnn_hybrid_pred) ** 2))
print(f"Pooled overall RMSE: {r_hybrid_all:.4f}")

# -- three-way comparison using the CORRECTED stratified_comparison_testsplit.npz
data = np.load('results/stratified_comparison_testsplit.npz')
cellnn_orig_true = data['cellnn_true']
cellnn_orig_pred = data['cellnn_pred']
mc_true = data['mc_true']
mc_pred = data['mc_pred']
anomaly = data['anomaly']

valid = ~np.isnan(anomaly)
abs_anom = np.abs(anomaly[valid])
terciles = np.percentile(abs_anom, [33.33, 66.67])
bin_labels = np.digitize(abs_anom, terciles)

cellnn_orig_true_v = cellnn_orig_true[valid]
cellnn_orig_pred_v = cellnn_orig_pred[valid]
mc_true_v = mc_true[valid]
mc_pred_v = mc_pred[valid]
cellnn_hybrid_true_v = cellnn_hybrid_true[valid]
cellnn_hybrid_pred_v = cellnn_hybrid_pred[valid]

names = ['Typical (bottom 33%)', 'Moderate (middle 33%)', 'Anomalous (top 33%)']
print(f"\n{'='*100}")
print("THREE-WAY (TEST SPLIT): Cell-NN (diffusion only) vs Cell-NN (+ climatology) vs Monte-Carlo")
print(f"{'='*100}")
print(f"{'Bin':<24}{'n':>9}{'CellNN-orig':>14}{'CellNN-hybrid':>16}{'MonteCarlo':>13}{'Best':>16}")
print("-" * 100)
for b in range(3):
    sel = bin_labels == b
    n = int(sel.sum())
    r_orig = np.sqrt(np.mean((cellnn_orig_true_v[sel] - cellnn_orig_pred_v[sel]) ** 2))
    r_hybrid = np.sqrt(np.mean((cellnn_hybrid_true_v[sel] - cellnn_hybrid_pred_v[sel]) ** 2))
    r_mc = np.sqrt(np.mean((mc_true_v[sel] - mc_pred_v[sel]) ** 2))
    best = min([('Cell-NN-orig', r_orig), ('Cell-NN-hybrid', r_hybrid), ('Monte-Carlo', r_mc)],
               key=lambda x: x[1])
    print(f"{names[b]:<24}{n:>9}{r_orig:>14.4f}{r_hybrid:>16.4f}{r_mc:>13.4f}{best[0]:>16}")

r_orig_all = np.sqrt(np.mean((cellnn_orig_true_v - cellnn_orig_pred_v) ** 2))
r_hybrid_all2 = np.sqrt(np.mean((cellnn_hybrid_true_v - cellnn_hybrid_pred_v) ** 2))
r_mc_all = np.sqrt(np.mean((mc_true_v - mc_pred_v) ** 2))
print("-" * 100)
print(f"{'OVERALL':<24}{len(abs_anom):>9}{r_orig_all:>14.4f}{r_hybrid_all2:>16.4f}{r_mc_all:>13.4f}")
print(f"{'='*100}")
print("\n[Compare to original Table 14: Typical=0.145/0.089/0.092, Moderate=0.145/0.127/0.099,")
print(" Anomalous=0.204/0.313/0.149, Overall=0.167/0.202/0.116 -- NOTE: original used kappa=beta=0.3,n=30]")

np.savez('results/cellnn_hybrid_comparison_testsplit.npz',
         cellnn_hybrid_true=cellnn_hybrid_true, cellnn_hybrid_pred=cellnn_hybrid_pred,
         rmse_per_t=rmse_per_t)
print("\nSaved -> results/cellnn_hybrid_comparison_testsplit.npz")
print("Done.")
