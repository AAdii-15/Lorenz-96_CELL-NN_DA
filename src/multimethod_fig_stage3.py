"""
Multi-method comparison figure -- Stage 3: Causal Monte Carlo
==================================================================
Fills REAL cloud gaps (not synthetic held-out pixels) for the 4
selected dates. Since the 4 dates only span 2017 and 2019, only two
distribution-fitting passes are needed (prior-to-2017, prior-to-2019),
not the full 5-year sweep the original validation script uses.
Saves both the ensemble mean and two individual stochastic draws.
"""
import numpy as np
from scipy import stats
from scipy.stats import kstest
import warnings
warnings.filterwarnings('ignore')

print("="*65)
print("Multi-method figure -- Stage 3: Causal Monte Carlo (real gaps)")
print("="*65)

chl_raw = np.load('data/modis_chl/chl_data.npy')
chl_norm = np.load('data/modis_chl/chl_norm.npy')
chl_mean = np.load('data/modis_chl/chl_mean.npy')
chl_std = np.load('data/modis_chl/chl_std.npy')
dates = np.load('data/modis_chl/chl_dates.npy')
N, LAT, LON = chl_raw.shape

date_strs = [str(d) for d in dates]
target_dates = ['20190829', '20190914', '20170501', '20170125']
target_idx = [date_strs.index(d) for d in target_dates]
years = np.array([int(str(int(d))[:4]) for d in dates])

ocean_mask = np.any(~np.isnan(chl_norm), axis=0)
obs_mask_all = ~np.isnan(chl_norm)   # real observations, no held-out masking

# ── Step I: spatial + backward temporal fill (real data, no masking) ──
print("\nStep I: spatial + backward temporal fill for target dates...")

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

step1_filled = {}
for d, idx in zip(target_dates, target_idx):
    f = chl_raw[idx].copy()
    f = neighbour_avg_axis(f, axis=0)
    f = neighbour_avg_axis(f, axis=1)

    gap = np.isnan(f)
    for back in range(1, min(idx, 10) + 1):
        if not gap.any():
            break
        cand = chl_raw[idx - back]
        usable = gap & ~np.isnan(cand)
        f[usable] = cand[usable]
        gap = gap & ~usable

    step1_filled[d] = f
    still_missing = (np.isnan(f) & ocean_mask).sum()
    print(f"  {d}: still missing after Step I: {still_missing}")

# ── Step II: per-pixel causal distribution fit (2 passes: 2017, 2019) ──
N_DRAWS = 10000
np.random.seed(7)

def fit_distributions_prior_to(year_threshold):
    print(f"\n  Fitting distributions using years < {year_threshold}...")
    prior_mask = years < year_threshold
    train_stack = np.where(obs_mask_all[prior_mask], chl_raw[prior_mask], np.nan)
    pixel_mean = np.full((LAT, LON), np.nan)
    pixel_draw1 = np.full((LAT, LON), np.nan)
    pixel_draw2 = np.full((LAT, LON), np.nan)
    n_fitted, n_kde, n_skipped = 0, 0, 0
    for i in range(LAT):
        if i % 100 == 0:
            print(f"    row {i}/{LAT}...")
        for j in range(LON):
            if not ocean_mask[i, j]:
                continue
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
            pixel_mean[i, j] = np.mean(draws)
            pixel_draw1[i, j] = draws[0]
            pixel_draw2[i, j] = draws[1] if len(draws) > 1 else draws[0]
    print(f"    Fitted: {n_fitted} parametric, {n_kde} KDE, {n_skipped} skipped")
    return pixel_mean, pixel_draw1, pixel_draw2

# Only fit for the two year-thresholds actually needed (2017, 2019)
fits = {}
for yr in sorted(set(years[i] for i in target_idx)):
    fits[yr] = fit_distributions_prior_to(yr)

# ── Combine Step I + Step II, per target date ──
mc_mean_results = {}
mc_draw1_results = {}
mc_draw2_results = {}

for d, idx in zip(target_dates, target_idx):
    yr = years[idx]
    p_mean, p_draw1, p_draw2 = fits[yr]

    still_nan = np.isnan(step1_filled[d]) & ocean_mask

    f_mean = step1_filled[d].copy()
    f_draw1 = step1_filled[d].copy()
    f_draw2 = step1_filled[d].copy()
    f_mean[still_nan] = p_mean[still_nan]
    f_draw1[still_nan] = p_draw1[still_nan]
    f_draw2[still_nan] = p_draw2[still_nan]

    # same-day fallback for anything still missing
    for f in (f_mean, f_draw1, f_draw2):
        gap = np.isnan(f) & ocean_mask
        if gap.sum() > 0:
            true_obs = chl_raw[idx][obs_mask_all[idx]]
            fallback = np.nanmean(true_obs) if true_obs.size > 0 else np.nan
            f[gap] = fallback

    mc_mean_results[d] = np.log10(np.clip(f_mean, 1e-6, None))
    mc_draw1_results[d] = np.log10(np.clip(f_draw1, 1e-6, None))
    mc_draw2_results[d] = np.log10(np.clip(f_draw2, 1e-6, None))
    print(f"  {d}: Monte Carlo reconstruction complete")

np.savez('results/multimethod_montecarlo.npz', target_dates=target_dates,
         **{f'mcmean_{d}': mc_mean_results[d] for d in target_dates},
         **{f'mcdraw1_{d}': mc_draw1_results[d] for d in target_dates},
         **{f'mcdraw2_{d}': mc_draw2_results[d] for d in target_dates})
print("\nSaved -> results/multimethod_montecarlo.npz")
print("Stage 3 done.")
