"""
VIIRS External Validation: Correct Protocol, κ=0.4, n_diff=50
===============================================================
Exact same protocol as validate_viirs_fixed2.py:
  - 20% MODIS pixels held out (seed=42, same stream as internal)
  - Reconstruction compared to VIIRS at held-out pixels
    where VIIRS is also valid
  - Scored in raw LOG10 space (not normalized)
    i.e. mu_a_log = mu_a * modis_std + modis_mean
         v_log    = v_field * viirs_std + viirs_mean
  - VIIRS loaded with its own mean/std (independent normalization)
Only change from original: kappa=0.4, n_diff=50 instead of 0.3/30.
"""
import numpy as np
import xarray as xr, glob, os
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("VIIRS External Validation: Correct Protocol")
print("kappa=0.4, n_diff=50 | Scored in log10 space")
print("=" * 65)

KAPPA  = 0.4
N_DIFF = 50

# ── MODIS ─────────────────────────────────────────────────────────────
chl_data    = np.load('data/modis_chl/chl_data.npy')
modis_dates = np.load('data/modis_chl/chl_dates.npy')
modis_log   = np.log10(chl_data)
modis_log[~np.isfinite(modis_log)] = np.nan
modis_mean  = float(np.nanmean(modis_log))
modis_std   = float(np.nanstd(modis_log))
modis_norm  = (modis_log - modis_mean) / modis_std
N, LAT, LON = modis_norm.shape
print(f"MODIS mean={modis_mean:.4f} std={modis_std:.4f}")

# ── VIIRS ─────────────────────────────────────────────────────────────
VIIRS_DIR   = "data/viirs_chl/nc_files"
viirs_files = sorted(glob.glob(f"{VIIRS_DIR}/*.nc"))
viirs_raw, viirs_dates_list = [], []
for f in viirs_files:
    ds = xr.open_dataset(f)
    viirs_raw.append(ds['chlor_a'].values)
    viirs_dates_list.append(os.path.basename(f).split('.')[1].split('_')[0])
    ds.close()
viirs_raw   = np.array(viirs_raw)
viirs_dates = np.array(viirs_dates_list)
viirs_log   = np.log10(viirs_raw)
viirs_log[~np.isfinite(viirs_log)] = np.nan
viirs_mean  = float(np.nanmean(viirs_log))
viirs_std   = float(np.nanstd(viirs_log))
viirs_norm  = (viirs_log - viirs_mean) / viirs_std
print(f"VIIRS mean={viirs_mean:.4f} std={viirs_std:.4f}")
print(f"Cross-sensor offset: {abs(modis_mean-viirs_mean):.4f}")

# ── Diffusion pipeline ─────────────────────────────────────────────────
def diffusion_pipeline(mu_b_2d, y_obs_2d, obs_mask_2d):
    mu_a = mu_b_2d.copy()
    mu_a[obs_mask_2d] = y_obs_2d[obs_mask_2d]
    for _ in range(N_DIFF):
        p = np.pad(mu_a, 1, mode='edge')
        L = p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*mu_a
        u = np.zeros_like(mu_a)
        u[~obs_mask_2d] = KAPPA * L[~obs_mask_2d]
        mu_a = mu_a + 0.1 * u
    return mu_a

# ── Run validation (identical protocol to validate_viirs_fixed2.py) ───
modis_date_set = {str(d): i for i,d in enumerate(modis_dates)}
viirs_date_set = {d: i for i,d in enumerate(viirs_dates)}
common_dates   = sorted(set(str(d) for d in modis_dates) &
                         set(viirs_dates))
print(f"\nCommon dates: {len(common_dates)}")

np.random.seed(42)   # CRITICAL: same seed as internal validation
rmse_cnn_list, rmse_bg_list = [], []

for t_idx, date in enumerate(common_dates[1:], 1):
    mi      = modis_date_set[date]
    vi      = viirs_date_set[date]
    mi_prev = modis_date_set[common_dates[t_idx-1]]

    m_curr  = modis_norm[mi]
    m_prev  = modis_norm[mi_prev]
    v_field = viirs_norm[vi]

    valid_m = ~np.isnan(m_curr)
    valid_v = ~np.isnan(v_field)
    if valid_m.sum() < 500:
        continue

    # Identical 20% mask protocol
    vidx = np.where(valid_m)
    n_v  = len(vidx[0])
    sel  = np.random.choice(n_v, int(0.2*n_v), replace=False)
    tr, tc = vidx[0][sel], vidx[1][sel]
    obs_mask = valid_m.copy()
    obs_mask[tr, tc] = False

    mu_b = m_prev.copy(); mu_b[np.isnan(mu_b)] = 0.0
    y_obs = np.where(obs_mask, m_curr, 0.0)

    mu_a = diffusion_pipeline(mu_b, y_obs, obs_mask)

    # Test pixels where VIIRS also valid
    t_mask = np.zeros_like(m_curr, dtype=bool)
    t_mask[tr, tc] = True
    c_mask = t_mask & valid_v
    if c_mask.sum() < 50:
        continue

    # Score in RAW LOG10 SPACE (not normalized) -- same as original
    mu_a_log = mu_a * modis_std + modis_mean
    mu_b_log = mu_b * modis_std + modis_mean
    v_log    = v_field * viirs_std + viirs_mean

    rmse_cnn_list.append(
        np.sqrt(np.mean((mu_a_log[c_mask]-v_log[c_mask])**2)))
    rmse_bg_list.append(
        np.sqrt(np.mean((mu_b_log[c_mask]-v_log[c_mask])**2)))

rmse_cnn = np.array(rmse_cnn_list)
rmse_bg  = np.array(rmse_bg_list)
improv   = (rmse_bg.mean()-rmse_cnn.mean())/rmse_bg.mean()*100

# Bootstrap CI
np.random.seed(42)
n = len(rmse_cnn)
improvs_boot = []
for _ in range(10000):
    idx = np.random.choice(n, n, replace=True)
    m_bg  = rmse_bg[idx].mean()
    m_cnn = rmse_cnn[idx].mean()
    improvs_boot.append((m_bg-m_cnn)/m_bg*100)
ci = np.percentile(improvs_boot, [2.5, 97.5])

print(f"\n{'='*65}")
print(f"VIIRS EXTERNAL VALIDATION RESULTS")
print(f"Protocol: held-out 20% mask, scored in log10 space")
print(f"{'='*65}")
print(f"Background RMSE : {rmse_bg.mean():.4f}")
print(f"Pipeline RMSE   : {rmse_cnn.mean():.4f}")
print(f"Improvement     : {improv:.1f}% [{ci[0]:.1f}%, {ci[1]:.1f}%]")
print(f"n timesteps     : {len(rmse_cnn)}")
print(f"\nSanity check vs original (κ=0.3, n_diff=30):")
print(f"  Original: BG=0.2394, CNN=0.0946, Improv=60.5%")
print(f"  New     : BG={rmse_bg.mean():.4f}, CNN={rmse_cnn.mean():.4f},"
      f" Improv={improv:.1f}%")
print(f"{'='*65}")

np.savez('results/viirs_optimal_correct.npz',
         rmse_cnn=rmse_cnn, rmse_bg=rmse_bg,
         improv=improv, ci=ci)
print("\nSaved -> results/viirs_optimal_correct.npz")
print("\nDone.")
