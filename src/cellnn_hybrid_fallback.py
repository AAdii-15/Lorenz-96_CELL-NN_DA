"""
Cell-NN with climatology-blended fallback -- testing the fix
================================================================
Diagnosis from the stratified analysis: Cell-NN's pure-diffusion
fallback for unobserved pixels has no memory of a pixel's own
identity/typical level, while Monte-Carlo's per-pixel climatology
does -- and won in every anomaly bin as a result.

Fix: add a single, FIXED (not tuned against the test set) pull term
toward each pixel's own historical climatological mean, alongside
the existing spatial diffusion, for unobserved pixels only. The
observed-pixel DA equation itself is completely untouched.

beta = kappa = 0.3 (equal weight to spatial smoothing and
climatological pull -- chosen for symmetry/simplicity, not tuned
to any specific result).
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath('.')))
from src.da_cellnn import cellnn_da_step
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Cell-NN with climatology-blended fallback (testing the fix)")
print("=" * 70)

chl_norm = np.load('data/modis_chl/chl_norm.npy')
dates = np.load('data/modis_chl/chl_dates.npy')

N, LAT, LON = chl_norm.shape

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
    n_test = int(0.2 * n_valid)
    sel = np.random.choice(n_valid, n_test, replace=False)
    test_rows = valid_idx[0][sel]
    test_cols = valid_idx[1][sel]
    test_mask[t, test_rows, test_cols] = True
    eval_t.append(t)

obs_mask_all = (~np.isnan(chl_norm)) & (~test_mask)
print(f"Held-out test pixels: {len(eval_t)} timesteps (identical to prior runs)")

clim_mean_2d = np.nanmean(np.where(obs_mask_all, chl_norm, np.nan), axis=0)
clim_valid = ~np.isnan(clim_mean_2d)
print(f"Climatology defined for {int(clim_valid.sum())}/{LAT*LON} pixels")

BETA = 0.3   # = kappa, fixed by symmetry, NOT tuned against test results

def cellnn_da_2d_hybrid(mu_b_2d, y_obs_2d, obs_mask_2d, alpha_A=1.0, wb=1.0,
                         kappa=0.3, beta=BETA, n_diff=30):
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

print("\nRunning Cell-NN (diffusion + climatology fallback) on all test timesteps...")

cellnn_hybrid_true, cellnn_hybrid_pred = [], []
rmse_per_t = []

for t in eval_t:
    field_curr = chl_norm[t]
    field_prev = chl_norm[t - 1]
    tm = test_mask[t]
    valid_mask = ~np.isnan(field_curr)

    mu_b = field_prev.copy()
    mu_b[np.isnan(mu_b)] = 0.0

    obs_mask = valid_mask & (~tm)
    y_obs = np.where(obs_mask, field_curr, 0.0)

    mu_a, n_iter = cellnn_da_2d_hybrid(mu_b, y_obs, obs_mask)

    rows, cols = np.where(tm)
    true_v = field_curr[rows, cols]
    pred_v = mu_a[rows, cols]
    cellnn_hybrid_true.append(true_v)
    cellnn_hybrid_pred.append(pred_v)

    rmse_per_t.append(np.sqrt(np.mean((true_v - pred_v) ** 2)))

cellnn_hybrid_true = np.concatenate(cellnn_hybrid_true)
cellnn_hybrid_pred = np.concatenate(cellnn_hybrid_pred)
rmse_per_t = np.array(rmse_per_t)

print(f"Done: {len(cellnn_hybrid_true)} test instances")
print(f"Mean RMSE (per-timestep avg, matches earlier headline format): {rmse_per_t.mean():.4f}")
r_hybrid_all = np.sqrt(np.mean((cellnn_hybrid_true - cellnn_hybrid_pred) ** 2))
print(f"Pooled overall RMSE (matches stratified script formula): {r_hybrid_all:.4f}")

# -- three-way comparison using already-saved Monte-Carlo + original Cell-NN
data = np.load('results/stratified_comparison.npz')
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
print("THREE-WAY: Cell-NN (diffusion only) vs Cell-NN (+ climatology) vs Monte-Carlo")
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

np.savez('results/cellnn_hybrid_comparison.npz',
         cellnn_hybrid_true=cellnn_hybrid_true, cellnn_hybrid_pred=cellnn_hybrid_pred,
         rmse_per_t=rmse_per_t)
print("\nSaved -> results/cellnn_hybrid_comparison.npz")
print("\nDone.")
