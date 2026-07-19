"""
CAUSAL Monte-Carlo Gap-Filling for MODIS Chl-a -- fair vs Cell-NN
====================================================================
The first run's per-pixel distributions were fit on the ENTIRE
2015-2019 record (including timesteps AFTER the one being tested) --
faithful to Modi et al. (2022)'s own retrospective design, but an
unfair information advantage vs Cell-NN, which only ever uses X(t-1).

This version: each pixel's distribution uses an EXPANDING WINDOW of
STRICTLY PRIOR YEARS only. Step I's temporal fill is backward-only.
2015 has no causal history at all -- expected, and itself a finding.
"""
import numpy as np
from scipy import stats
from scipy.stats import kstest
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CAUSAL Monte-Carlo Gap-Filling -- MODIS Chl-a (fair vs Cell-NN)")
print("=" * 70)

chl_raw = np.load('data/modis_chl/chl_data.npy')
chl_norm = np.load('data/modis_chl/chl_norm.npy')
chl_mean = np.load('data/modis_chl/chl_mean.npy')
chl_std = np.load('data/modis_chl/chl_std.npy')
dates = np.load('data/modis_chl/chl_dates.npy')

N, LAT, LON = chl_raw.shape
print(f"Shape: {chl_raw.shape}")

years = np.array([int(str(int(d))[:4]) for d in dates])
unique_years = sorted(set(years.tolist()))
print(f"Years present: {unique_years}")

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
print(f"Held-out test pixels: {len(eval_t)} timesteps, "
      f"{total_test_pixels} test pixel-instances (identical set as before)")

obs_mask_all = (~np.isnan(chl_norm)) & (~test_mask)

print("\nStep I: lon -> lat spatial interpolation, BACKWARD-ONLY time fill...")
filled_raw = chl_raw.copy()
filled_raw[test_mask] = np.nan

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

n_after_step1 = int(np.isnan(filled_raw)[test_mask].sum())
print(f"Test pixels still missing after Step I: {n_after_step1}/{total_test_pixels}")

print("\nStep II: per-pixel CAUSAL distribution fit (expanding window, prior years only)")
print("(refit per pixel at each year boundary -- this is the slow part)")

N_DRAWS = 10000
still_nan = np.isnan(filled_raw)

for Y in unique_years:
    t_in_year = [t for t in eval_t if years[t] == Y]
    if len(t_in_year) == 0:
        continue
    print(f"\n  Year {Y}: {len(t_in_year)} test timesteps")
    prior_mask = years < Y
    pixel_mc_mean = np.full((LAT, LON), np.nan)

    if prior_mask.sum() == 0:
        print("    No prior years available -- all gaps this year use spatial fallback")
    else:
        train_stack = np.where(obs_mask_all[prior_mask], chl_raw[prior_mask], np.nan)
        n_fitted, n_kde, n_skipped = 0, 0, 0
        for i in range(LAT):
            if i % 100 == 0:
                print(f"    row {i}/{LAT}...")
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
        print(f"    Fitted: {n_fitted} parametric, {n_kde} KDE, {n_skipped} skipped "
              f"(trained on years {sorted(set(years[prior_mask].tolist()))})")

    for t in t_in_year:
        mask_t = still_nan[t]
        filled_raw[t][mask_t] = pixel_mc_mean[mask_t]

unfilled_test = int(np.isnan(filled_raw)[test_mask].sum())
coverage_pct = 100 * (1 - unfilled_test / total_test_pixels)
print(f"\nCOVERAGE CHECK: {unfilled_test}/{total_test_pixels} still NaN "
      f"({coverage_pct:.2f}% covered before fallback)")

if unfilled_test > 0:
    print("Applying same-day basin-wide-mean fallback for remaining gaps...")
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
rmse_mc_by_year, rmse_bg_by_year = {}, {}
for t in eval_t:
    tm = test_mask[t]
    if tm.sum() == 0:
        continue
    true_v = chl_norm[t][tm]
    mc_v = norm_filled[t][tm]
    bg_v = chl_norm[t - 1][tm]
    bg_v = np.where(np.isnan(bg_v), 0.0, bg_v)
    r_mc = np.sqrt(np.mean((true_v - mc_v) ** 2))
    r_bg = np.sqrt(np.mean((true_v - bg_v) ** 2))
    rmse_mc.append(r_mc)
    rmse_bg.append(r_bg)
    y = years[t]
    rmse_mc_by_year.setdefault(y, []).append(r_mc)
    rmse_bg_by_year.setdefault(y, []).append(r_bg)

rmse_mc = np.array(rmse_mc)
rmse_bg = np.array(rmse_bg)
improvement = (rmse_bg.mean() - rmse_mc.mean()) / rmse_bg.mean() * 100

print(f"\n{'=' * 70}")
print("FINAL RESULTS -- CAUSAL Monte-Carlo (fair information constraint)")
print(f"{'=' * 70}")
print(f"{'Metric':<30} {'Monte-Carlo':>14} {'BG(t-1)':>12}")
print(f"{'-' * 70}")
print(f"{'Mean RMSE (all years)':<30} {rmse_mc.mean():>14.4f} {rmse_bg.mean():>12.4f}")
print(f"{'Improvement vs BG':<30} {improvement:>13.1f}%")
print(f"{'-' * 70}")
print("RMSE by year (shows the cost of having no/little causal history):")
for y in sorted(rmse_mc_by_year.keys()):
    mc_y = np.mean(rmse_mc_by_year[y])
    print(f"  {y}: Monte-Carlo RMSE = {mc_y:.4f}  (n={len(rmse_mc_by_year[y])} timesteps)")
print(f"{'=' * 70}")

np.save('results/montecarlo_causal_rmse.npy', rmse_mc)
np.save('results/montecarlo_causal_rmse_bg.npy', rmse_bg)
print("\nSaved -> results/montecarlo_causal_rmse.npy")
print("Saved -> results/montecarlo_causal_rmse_bg.npy")

try:
    rmse_cnn = np.load('results/chl_proper_rmse_cellnn.npy')
    print(f"\n{'Cell-NN mean RMSE':<30} {rmse_cnn.mean():>14.4f}")
    print(f"{'Causal Monte-Carlo mean RMSE':<30} {rmse_mc.mean():>14.4f}")
    if rmse_cnn.mean() < rmse_mc.mean():
        diff = (rmse_mc.mean() - rmse_cnn.mean()) / rmse_mc.mean() * 100
        print(f"Cell-NN is {diff:.1f}% lower RMSE than causal Monte-Carlo")
    else:
        diff = (rmse_cnn.mean() - rmse_mc.mean()) / rmse_cnn.mean() * 100
        print(f"Causal Monte-Carlo is {diff:.1f}% lower RMSE than Cell-NN")
except FileNotFoundError:
    print("\n(Cell-NN RMSE file not found)")

print("\nDone.")
