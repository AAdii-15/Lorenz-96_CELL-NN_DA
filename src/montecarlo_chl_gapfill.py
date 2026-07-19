"""
Monte-Carlo Gap-Filling for MODIS Chl-a (Bay of Bengal) -- CORRECTED
=========================================================
Faithful implementation of Modi, Roxy & Ghosh (2022, Sci. Rep.), same
protocol as before, PLUS a fallback fill + explicit coverage check so
RMSE is computed on the FULL held-out test set (not silently dropping
pixels Step II failed to reach), matching how Cell-NN is evaluated.
"""
import numpy as np
from scipy import stats
from scipy.stats import kstest
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("Monte-Carlo Gap-Filling -- MODIS Chl-a (Modi et al. 2022 method)")
print("=" * 65)

chl_raw = np.load('data/modis_chl/chl_data.npy')
chl_norm = np.load('data/modis_chl/chl_norm.npy')
chl_mean = np.load('data/modis_chl/chl_mean.npy')
chl_std = np.load('data/modis_chl/chl_std.npy')
dates = np.load('data/modis_chl/chl_dates.npy')

N, LAT, LON = chl_raw.shape
print(f"Shape: {chl_raw.shape}")

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

total_test_pixels = int(test_mask.sum())
print(f"Held-out test pixels reproduced for {len(eval_t)} timesteps "
      f"({total_test_pixels} test pixel-instances total)")

obs_mask_all = (~np.isnan(chl_norm)) & (~test_mask)

print("\nStep I: optimal linear interpolation (lon -> lat -> time)...")
filled_raw = chl_raw.copy()
filled_raw[test_mask] = np.nan

def neighbour_avg_axis(field, axis):
    out = field.copy()
    nan_mask = np.isnan(field)
    if axis == 0:
        left = np.roll(field, 1, axis=1)
        right = np.roll(field, -1, axis=1)
    else:
        left = np.roll(field, 1, axis=0)
        right = np.roll(field, -1, axis=0)
    l_valid = ~np.isnan(left)
    r_valid = ~np.isnan(right)
    both = l_valid & r_valid
    only_l = l_valid & ~r_valid
    only_r = r_valid & ~l_valid
    avg = np.full_like(field, np.nan)
    avg[both] = (left[both] + right[both]) / 2.0
    avg[only_l] = left[only_l]
    avg[only_r] = right[only_r]
    out[nan_mask] = avg[nan_mask]
    return out

for t in eval_t:
    f = filled_raw[t]
    f = neighbour_avg_axis(f, axis=0)
    f = neighbour_avg_axis(f, axis=1)
    filled_raw[t] = f

still_nan = np.isnan(filled_raw)
prev_val = np.roll(filled_raw, 1, axis=0)
next_val = np.roll(filled_raw, -1, axis=0)
prev_valid = ~np.isnan(prev_val)
next_valid = ~np.isnan(next_val)
both = prev_valid & next_valid
onlyp = prev_valid & ~next_valid
onlyn = next_valid & ~prev_valid
time_avg = np.full_like(filled_raw, np.nan)
time_avg[both] = (prev_val[both] + next_val[both]) / 2.0
time_avg[onlyp] = prev_val[onlyp]
time_avg[onlyn] = next_val[onlyn]
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

print(f"Parametric fit used for {n_fitted} pixels, "
      f"KDE fallback for {n_kde}, skipped (insufficient history) {n_skipped}")

still_nan = np.isnan(filled_raw)
for t in eval_t:
    mask_t = still_nan[t]
    filled_raw[t][mask_t] = pixel_mc_mean[mask_t]

# -- NEW: explicit coverage check + fallback for any STILL-missing
#         held-out test pixel (basin-wide mean of true observed
#         pixels for that exact timestep) so every test pixel gets
#         scored, matching Cell-NN's 100%-coverage evaluation.
unfilled_test = int(np.isnan(filled_raw)[test_mask].sum())
coverage_pct = 100 * (1 - unfilled_test / total_test_pixels)
print(f"\nCOVERAGE CHECK: {unfilled_test}/{total_test_pixels} held-out "
      f"test pixels still NaN after Step II ({coverage_pct:.2f}% covered)")

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
print(f"After fallback: {remaining}/{total_test_pixels} test pixels still NaN "
      f"(should be 0)")

log_filled = np.log10(np.clip(filled_raw, 1e-6, None))
norm_filled = (log_filled - chl_mean) / chl_std

rmse_mc, rmse_bg = [], []
for t in eval_t:
    tm = test_mask[t]
    if tm.sum() == 0:
        continue
    true_v = chl_norm[t][tm]
    mc_v = norm_filled[t][tm]
    bg_v = chl_norm[t - 1][tm]
    bg_v = np.where(np.isnan(bg_v), 0.0, bg_v)
    rmse_mc.append(np.sqrt(np.mean((true_v - mc_v) ** 2)))   # plain mean, no nanmean
    rmse_bg.append(np.sqrt(np.mean((true_v - bg_v) ** 2)))

rmse_mc = np.array(rmse_mc)
rmse_bg = np.array(rmse_bg)
improvement = (rmse_bg.mean() - rmse_mc.mean()) / rmse_bg.mean() * 100

print(f"\n{'=' * 65}")
print("FINAL RESULTS -- Monte-Carlo (Modi et al. 2022 method), FULL COVERAGE")
print(f"{'=' * 65}")
print(f"{'Metric':<30} {'Monte-Carlo':>14} {'BG(t-1)':>12}")
print(f"{'-' * 65}")
print(f"{'Mean RMSE':<30} {rmse_mc.mean():>14.4f} {rmse_bg.mean():>12.4f}")
print(f"{'Improvement vs BG':<30} {improvement:>13.1f}%")
print(f"{'Test pixel coverage':<30} {coverage_pct:>13.2f}%")
print(f"{'=' * 65}")

np.save('results/montecarlo_rmse.npy', rmse_mc)
np.save('results/montecarlo_rmse_bg.npy', rmse_bg)
print("\nSaved -> results/montecarlo_rmse.npy")
print("Saved -> results/montecarlo_rmse_bg.npy")

try:
    rmse_cnn = np.load('results/chl_proper_rmse_cellnn.npy')
    print(f"\n{'Cell-NN mean RMSE':<30} {rmse_cnn.mean():>14.4f}")
    print(f"{'Monte-Carlo mean RMSE':<30} {rmse_mc.mean():>14.4f}")
    if rmse_cnn.mean() < rmse_mc.mean():
        diff = (rmse_mc.mean() - rmse_cnn.mean()) / rmse_mc.mean() * 100
        print(f"Cell-NN is {diff:.1f}% lower RMSE than Monte-Carlo")
    else:
        diff = (rmse_cnn.mean() - rmse_mc.mean()) / rmse_cnn.mean() * 100
        print(f"Monte-Carlo is {diff:.1f}% lower RMSE than Cell-NN")
except FileNotFoundError:
    print("\n(Cell-NN RMSE file not found)")

print("\nDone.")
