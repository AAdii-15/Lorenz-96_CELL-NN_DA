"""
Stratified comparison -- TEST SPLIT (leakage fix) + kappa correction
========================================================================
Two fixes from the original stratified_comparison.py:
  1. kappa=0.4, n_diff=50 now used explicitly (was hardcoded to the
     stale kappa=0.3, n_diff=30 -- contradicts the paper's own stated
     "all later results use kappa=0.4, n=50", confirmed with user).
  2. obs_mask excludes the FULL held_out_20 (unchanged reconstruction
     input); scoring and anomaly-tercile binning are against
     test_mask only, not the full original 20%.
Causal Monte-Carlo logic (Part B) is UNCHANGED -- it was never tuned
against held-out pixels, so no leakage there; only which pixels it's
scored against moves.
"""
import numpy as np
from scipy import stats
from scipy.stats import kstest
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.da_cellnn import cellnn_da_step
import warnings
warnings.filterwarnings('ignore')

chl_raw  = np.load('data/modis_chl/chl_data.npy')
chl_norm = np.load('data/modis_chl/chl_norm.npy')
chl_mean = np.load('data/modis_chl/chl_mean.npy')
chl_std  = np.load('data/modis_chl/chl_std.npy')
dates    = np.load('data/modis_chl/chl_dates.npy')
N, LAT, LON = chl_norm.shape
years = np.array([int(str(int(d))[:4]) for d in dates])
unique_years = sorted(set(years.tolist()))

split = np.load('results/holdout_split_masks.npz')
held_out_20 = split['held_out_20']
test_mask   = split['test_mask']
eval_t      = list(split['eval_t'])

obs_mask_all = (~np.isnan(chl_norm)) & (~held_out_20)
print(f"Test pixels: {int(test_mask.sum()):,} across {len(eval_t)} timesteps")

clim_mean = np.nanmean(np.where(obs_mask_all, chl_norm, np.nan), axis=0)

# ============================================================
# PART A: Cell-NN -- kappa=0.4, n_diff=50 (corrected)
# ============================================================
print("\nPart A: re-running Cell-NN DA (kappa=0.4, n_diff=50), test predictions...")

def cellnn_da_2d(mu_b_2d, y_obs_2d, obs_mask_2d, alpha_A=1.0, wb=1.0,
                  kappa=0.4, n_diff=50):
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
        update = np.zeros_like(mu_a_2d)
        update[~obs_mask_2d] = kappa * laplacian[~obs_mask_2d]
        mu_a_2d = mu_a_2d + 0.1 * update
    return mu_a_2d, n_iter

cellnn_true_vals, cellnn_pred_vals, cellnn_anom_vals = [], [], []

for t in eval_t:
    field_curr = chl_norm[t]
    field_prev = chl_norm[t - 1]
    tm = test_mask[t]
    valid_mask = ~np.isnan(field_curr)
    mu_b = field_prev.copy(); mu_b[np.isnan(mu_b)] = 0.0
    obs_mask = valid_mask & (~held_out_20[t])
    y_obs = np.where(obs_mask, field_curr, 0.0)

    mu_a, n_iter = cellnn_da_2d(mu_b, y_obs, obs_mask,
                                 alpha_A=1.0, wb=1.0, kappa=0.4, n_diff=50)

    rows, cols = np.where(tm)
    if len(rows) == 0:
        continue
    cellnn_true_vals.append(field_curr[rows, cols])
    cellnn_pred_vals.append(mu_a[rows, cols])
    cellnn_anom_vals.append(field_curr[rows, cols] - clim_mean[rows, cols])

cellnn_true_vals = np.concatenate(cellnn_true_vals)
cellnn_pred_vals = np.concatenate(cellnn_pred_vals)
cellnn_anom_vals = np.concatenate(cellnn_anom_vals)
print(f"Cell-NN: {len(cellnn_true_vals)} test instances processed")

# ============================================================
# PART B: Causal Monte-Carlo -- UNCHANGED logic, test_mask scoring
# ============================================================
print("\nPart B: re-running causal Monte-Carlo, test predictions...")

filled_raw = chl_raw.copy()
filled_raw[held_out_20] = np.nan

def neighbour_avg_axis(field, axis):
    out = field.copy()
    nan_mask = np.isnan(field)
    if axis == 0:
        left = np.roll(field, 1, axis=1); right = np.roll(field, -1, axis=1)
    else:
        left = np.roll(field, 1, axis=0); right = np.roll(field, -1, axis=0)
    l_valid = ~np.isnan(left); r_valid = ~np.isnan(right)
    both = l_valid & r_valid; only_l = l_valid & ~r_valid; only_r = r_valid & ~l_valid
    avg = np.full_like(field, np.nan)
    avg[both] = (left[both] + right[both]) / 2.0
    avg[only_l] = left[only_l]; avg[only_r] = right[only_r]
    out[nan_mask] = avg[nan_mask]
    return out

for t in eval_t:
    f = filled_raw[t]
    f = neighbour_avg_axis(f, axis=0)
    f = neighbour_avg_axis(f, axis=1)
    filled_raw[t] = f

for t in eval_t:
    gap = np.isnan(filled_raw[t])
    if gap.sum() == 0:
        continue
    for back in range(1, min(t, 10) + 1):
        if not gap.any():
            break
        cand = filled_raw[t - back]
        usable = gap & ~np.isnan(cand)
        filled_raw[t][usable] = cand[usable]
        gap = gap & ~usable

N_DRAWS = 10000
still_nan = np.isnan(filled_raw)

for Y in unique_years:
    t_in_year = [t for t in eval_t if years[t] == Y]
    if len(t_in_year) == 0:
        continue
    print(f"  Year {Y}: {len(t_in_year)} test timesteps")
    prior_mask = years < Y
    pixel_mc_mean = np.full((LAT, LON), np.nan)
    if prior_mask.sum() > 0:
        train_stack = np.where(obs_mask_all[prior_mask], chl_raw[prior_mask], np.nan)
        for i in range(LAT):
            if i % 100 == 0:
                print(f"    row {i}/{LAT}")
            for j in range(LON):
                series = train_stack[:, i, j]
                vals = series[~np.isnan(series)]
                if len(vals) < 10:
                    continue
                best_dist, best_p, best_params = None, 0.0, None
                for dist_name in ['norm', 'lognorm', 'gamma']:
                    dist = getattr(stats, dist_name)
                    try:
                        params = dist.fit(vals)
                        D, p = kstest(vals, dist_name, args=params)
                        if p > best_p:
                            best_p, best_dist, best_params = p, dist_name, params
                    except Exception:
                        continue
                if best_p >= 0.05 and best_dist is not None:
                    dist = getattr(stats, best_dist)
                    draws = dist.rvs(*best_params, size=N_DRAWS)
                else:
                    try:
                        kde = stats.gaussian_kde(vals)
                        draws = kde.resample(N_DRAWS)[0]
                    except Exception:
                        draws = vals
                pixel_mc_mean[i, j] = np.mean(draws)
    for t in t_in_year:
        mask_t = still_nan[t]
        filled_raw[t][mask_t] = pixel_mc_mean[mask_t]

for t in eval_t:
    tm = test_mask[t]
    gap = tm & np.isnan(filled_raw[t])
    if gap.sum() == 0:
        continue
    true_obs_t = chl_raw[t][obs_mask_all[t]]
    fallback_val = np.nanmean(true_obs_t) if true_obs_t.size > 0 else np.nan
    filled_raw[t][gap] = fallback_val

log_filled = np.log10(np.clip(filled_raw, 1e-6, None))
norm_filled = (log_filled - chl_mean) / chl_std

mc_true_vals, mc_pred_vals, mc_anom_vals = [], [], []
for t in eval_t:
    tm = test_mask[t]
    rows, cols = np.where(tm)
    if len(rows) == 0:
        continue
    mc_true_vals.append(chl_norm[t][rows, cols])
    mc_pred_vals.append(norm_filled[t][rows, cols])
    mc_anom_vals.append(chl_norm[t][rows, cols] - clim_mean[rows, cols])

mc_true_vals = np.concatenate(mc_true_vals)
mc_pred_vals = np.concatenate(mc_pred_vals)
mc_anom_vals = np.concatenate(mc_anom_vals)
print(f"Monte-Carlo: {len(mc_true_vals)} test instances processed")

# ============================================================
# PART C: Stratify by |anomaly| terciles (test set only)
# ============================================================
print("\nPart C: stratifying by |anomaly| terciles...")

abs_anom = np.abs(cellnn_anom_vals)
terciles = np.nanpercentile(abs_anom, [33.33, 66.67])
bin_labels = np.digitize(abs_anom, terciles)

names = ['Typical (bottom 33%)', 'Moderate (middle 33%)', 'Anomalous (top 33%)']
print(f"\n{'Bin':<26}{'n':>10}{'Cell-NN RMSE':>16}{'Monte-Carlo RMSE':>20}{'Winner':>14}")
print("-" * 86)
for b in range(3):
    sel = bin_labels == b
    n = int(sel.sum())
    r_cnn = np.sqrt(np.mean((cellnn_true_vals[sel] - cellnn_pred_vals[sel]) ** 2))
    r_mc = np.sqrt(np.mean((mc_true_vals[sel] - mc_pred_vals[sel]) ** 2))
    winner = "Cell-NN" if r_cnn < r_mc else "Monte-Carlo"
    print(f"{names[b]:<26}{n:>10}{r_cnn:>16.4f}{r_mc:>20.4f}{winner:>14}")

r_cnn_all = np.sqrt(np.mean((cellnn_true_vals - cellnn_pred_vals) ** 2))
r_mc_all = np.sqrt(np.mean((mc_true_vals - mc_pred_vals) ** 2))
print("-" * 86)
print(f"{'OVERALL':<26}{len(abs_anom):>10}{r_cnn_all:>16.4f}{r_mc_all:>20.4f}")
print("\n[Compare to original Table 13: Typical=0.145/0.092, Moderate=0.145/0.099,")
print(" Anomalous=0.204/0.149, Overall=0.167/0.116 -- NOTE: original used kappa=0.3,n=30]")

np.savez('results/stratified_comparison_testsplit.npz',
         cellnn_true=cellnn_true_vals, cellnn_pred=cellnn_pred_vals,
         mc_true=mc_true_vals, mc_pred=mc_pred_vals,
         anomaly=cellnn_anom_vals, bin_labels=bin_labels)
print("\nSaved -> results/stratified_comparison_testsplit.npz")
print("Done.")
