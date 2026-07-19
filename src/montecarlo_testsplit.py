"""
Monte-Carlo Gap-Filling -- TEST SPLIT (leakage fix)
========================================================
Identical Step I / Step II logic to src/montecarlo_chl_gapfill.py.
Only change: excludes the FULL held_out_20 (not just the old 20%
mask) from per-pixel distribution fitting -- same set, loaded from
the canonical split file for consistency -- and scores final RMSE
only against test_mask (the untouched 10% half).
"""
import numpy as np
from scipy import stats
from scipy.stats import kstest
import warnings
warnings.filterwarnings('ignore')

chl_raw  = np.load('data/modis_chl/chl_data.npy')
chl_norm = np.load('data/modis_chl/chl_norm.npy')
chl_mean = np.load('data/modis_chl/chl_mean.npy')
chl_std  = np.load('data/modis_chl/chl_std.npy')
dates    = np.load('data/modis_chl/chl_dates.npy')
N, LAT, LON = chl_raw.shape

split = np.load('results/holdout_split_masks.npz')
held_out_20 = split['held_out_20']
test_mask   = split['test_mask']
eval_t      = list(split['eval_t'])

total_test_pixels = int(test_mask.sum())
print(f"Held-out (excluded from fitting): {int(held_out_20.sum())}")
print(f"Test pixels (scored):             {total_test_pixels}")

obs_mask_all = (~np.isnan(chl_norm)) & (~held_out_20)

print("\nStep I: optimal linear interpolation (lon -> lat -> time)...")
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

still_nan = np.isnan(filled_raw)
prev_val = np.roll(filled_raw, 1, axis=0); next_val = np.roll(filled_raw, -1, axis=0)
prev_valid = ~np.isnan(prev_val); next_valid = ~np.isnan(next_val)
both = prev_valid & next_valid; onlyp = prev_valid & ~next_valid; onlyn = next_valid & ~prev_valid
time_avg = np.full_like(filled_raw, np.nan)
time_avg[both] = (prev_val[both] + next_val[both]) / 2.0
time_avg[onlyp] = prev_val[onlyp]; time_avg[onlyn] = next_val[onlyn]
filled_raw[still_nan] = time_avg[still_nan]

print("\nStep II: per-pixel distribution fit + N=10,000 Monte-Carlo draws...")
print("(runs ONCE per pixel -- this may take several minutes)")

N_DRAWS = 10000
pixel_mc_mean = np.full((LAT, LON), np.nan)
train_stack = np.where(obs_mask_all, chl_raw, np.nan)

n_fitted, n_kde, n_skipped = 0, 0, 0
for i in range(LAT):
    if i % 50 == 0:
        print(f"  row {i}/{LAT}...")
    for j in range(LON):
        series = train_stack[:, i, j]
        vals = series[~np.isnan(series)]
        if len(vals) < 10:
            n_skipped += 1
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
            n_fitted += 1
        else:
            try:
                kde = stats.gaussian_kde(vals)
                draws = kde.resample(N_DRAWS)[0]
                n_kde += 1
            except Exception:
                draws = vals
        pixel_mc_mean[i, j] = np.mean(draws)

print(f"Parametric fit used for {n_fitted} pixels, KDE fallback for {n_kde}, skipped {n_skipped}")

still_nan = np.isnan(filled_raw)
for t in eval_t:
    mask_t = still_nan[t]
    filled_raw[t][mask_t] = pixel_mc_mean[mask_t]

unfilled_test = int(np.isnan(filled_raw)[test_mask].sum())
coverage_pct = 100 * (1 - unfilled_test / total_test_pixels)
print(f"\nCOVERAGE CHECK: {unfilled_test}/{total_test_pixels} test pixels still NaN ({coverage_pct:.2f}% covered)")

if unfilled_test > 0:
    print("Applying basin-wide-mean fallback for uncovered test pixels...")
    for t in eval_t:
        tm = test_mask[t]
        gap = tm & np.isnan(filled_raw[t])
        if gap.sum() == 0:
            continue
        true_obs_t = chl_raw[t][obs_mask_all[t]]
        fallback_val = np.nanmean(true_obs_t) if true_obs_t.size > 0 else np.nan
        filled_raw[t][gap] = fallback_val

remaining = int(np.isnan(filled_raw)[test_mask].sum())
print(f"After fallback: {remaining}/{total_test_pixels} still NaN (should be 0)")

log_filled = np.log10(np.clip(filled_raw, 1e-6, None))
norm_filled = (log_filled - chl_mean) / chl_std

rmse_mc, rmse_bg = [], []
for t in eval_t:
    tm = test_mask[t]
    if tm.sum() == 0:
        continue
    true_v = chl_norm[t][tm]
    mc_v = norm_filled[t][tm]
    bg_v = chl_norm[t-1][tm]
    bg_v = np.where(np.isnan(bg_v), 0.0, bg_v)
    rmse_mc.append(np.sqrt(np.mean((true_v - mc_v) ** 2)))
    rmse_bg.append(np.sqrt(np.mean((true_v - bg_v) ** 2)))

rmse_mc = np.array(rmse_mc); rmse_bg = np.array(rmse_bg)
improvement = (rmse_bg.mean() - rmse_mc.mean()) / rmse_bg.mean() * 100
print(f"\nBackground RMSE (test-only): {rmse_bg.mean():.4f}")
print(f"Monte-Carlo RMSE (test-only): {rmse_mc.mean():.4f}")
print(f"Improvement: {improvement:.1f}%  [was RMSE=0.120, 84.1%]")
print(f"Test pixel coverage: {coverage_pct:.2f}%")

np.save('results/montecarlo_rmse_testsplit.npy', rmse_mc)
np.save('results/montecarlo_rmse_bg_testsplit.npy', rmse_bg)
print("\nSaved -> results/montecarlo_rmse_testsplit.npy")
print("Saved -> results/montecarlo_rmse_bg_testsplit.npy")
print("Done.")
