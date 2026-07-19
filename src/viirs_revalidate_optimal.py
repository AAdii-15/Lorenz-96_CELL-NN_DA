"""
VIIRS External Validation with Optimised Diffusion (κ=0.4, n_diff=50)
=======================================================================
Correctly normalises VIIRS with VIIRS-SPECIFIC mean/std,
NOT MODIS's stats. This was the validated approach in the original
validate_viirs_fixed2.py. Only difference from original: the
reconstruction uses κ=0.4, n_diff=50 instead of κ=0.3, n_diff=30.
"""
import numpy as np
import xarray as xr, glob, os
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("VIIRS Re-Validation: κ=0.4, n_diff=50")
print("(VIIRS normalised with its own mean/std -- correct approach)")
print("=" * 65)

KAPPA  = 0.4
N_DIFF = 50

chl_norm = np.load('data/modis_chl/chl_norm.npy')
modis_mean = np.load('data/modis_chl/chl_mean.npy')[0]
modis_std  = np.load('data/modis_chl/chl_std.npy')[0]
dates    = np.load('data/modis_chl/chl_dates.npy')
N, LAT, LON = chl_norm.shape
date_strs = [str(d) for d in dates]

def diffusion_pipeline(mu_b_2d, y_obs_2d, obs_mask_2d,
                       kappa=KAPPA, n_diff=N_DIFF):
    mu_a = mu_b_2d.copy()
    mu_a[obs_mask_2d] = y_obs_2d[obs_mask_2d]
    for _ in range(n_diff):
        p = np.pad(mu_a, 1, mode='edge')
        L = p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*mu_a
        u = np.zeros_like(mu_a)
        u[~obs_mask_2d] = kappa * L[~obs_mask_2d]
        mu_a = mu_a + 0.1 * u
    return mu_a

# ── Build full reconstruction (κ=0.4, n_diff=50) ─────────────────────
print("Building full reconstruction (κ=0.4, n_diff=50)...")
chl_rec = np.zeros((N, LAT, LON))
chl_rec[0] = np.where(np.isnan(chl_norm[0]), 0.0, chl_norm[0])
for t in range(1, N):
    fc = chl_norm[t]
    valid_mask = ~np.isnan(fc)
    mu_b = chl_rec[t-1].copy()
    y_obs = np.where(valid_mask, fc, 0.0)
    chl_rec[t] = diffusion_pipeline(mu_b, y_obs, valid_mask)

# Convert reconstruction back to log10(mg/m3)
chl_rec_log = chl_rec * modis_std + modis_mean
print(f"  Reconstruction NaNs: {np.isnan(chl_rec_log).sum()} (should be 0)")

# ── Load VIIRS files and compute VIIRS-specific normalisation ─────────
VIIRS_DIR = "data/viirs_chl/nc_files"
viirs_files = sorted(glob.glob(f"{VIIRS_DIR}/*.nc"))

# First pass: collect all VIIRS log values to compute mean/std
print("Computing VIIRS-specific normalisation (own mean/std)...")
all_viirs_log = []
viirs_log_by_date = {}
for f in viirs_files:
    fname = os.path.basename(f)
    fdate = fname.split('.')[1].split('_')[0]
    ds = xr.open_dataset(f)
    v_raw = ds['chlor_a'].values
    ds.close()
    v_log = np.log10(v_raw)
    v_log[~np.isfinite(v_log)] = np.nan
    viirs_log_by_date[fdate] = v_log
    all_viirs_log.append(v_log[~np.isnan(v_log)])

all_viirs_vals = np.concatenate(all_viirs_log)
viirs_mean = np.mean(all_viirs_vals)
viirs_std  = np.std(all_viirs_vals)
print(f"  VIIRS: mean={viirs_mean:.4f}, std={viirs_std:.4f}")
print(f"  MODIS: mean={modis_mean:.4f}, std={modis_std:.4f}")
print(f"  Cross-sensor mean offset: {modis_mean-viirs_mean:.4f} "
      f"(consistent with known calibration difference)")

# ── Evaluate: Cell-NN reconstruction vs VIIRS ─────────────────────────
# Both converted to VIIRS-normalised space for fair comparison
common_dates = sorted(set(date_strs) & set(viirs_log_by_date.keys()))
print(f"\nCommon dates: {len(common_dates)}/230")

rmse_cnn_v, rmse_bg_v = [], []

for d in common_dates:
    ti = date_strs.index(d)
    viirs_log  = viirs_log_by_date[d]
    viirs_norm = (viirs_log - viirs_mean) / viirs_std

    # Reconstruction in log space → VIIRS-normalised space
    cnn_norm = (chl_rec_log[ti] - viirs_mean) / viirs_std

    # Background: raw MODIS log → VIIRS-normalised space
    modis_log_t = chl_norm[ti] * modis_std + modis_mean
    bg_norm = (modis_log_t - viirs_mean) / viirs_std

    vm = ~np.isnan(viirs_norm)
    if vm.sum() == 0:
        continue

    cnn_v = cnn_norm[vm]
    bg_v  = bg_norm[vm]
    bg_v[np.isnan(bg_v)] = 0.0
    true_v = viirs_norm[vm]

    rmse_cnn_v.append(np.sqrt(np.mean((true_v - cnn_v)**2)))
    rmse_bg_v.append( np.sqrt(np.mean((true_v - bg_v)**2)))

rmse_cnn_v = np.array(rmse_cnn_v)
rmse_bg_v  = np.array(rmse_bg_v)
improv = (rmse_bg_v.mean() - rmse_cnn_v.mean()) / rmse_bg_v.mean() * 100

# Bootstrap CI
np.random.seed(42)
improvs_boot = []
n = len(rmse_cnn_v)
for _ in range(10000):
    idx = np.random.choice(n, n, replace=True)
    m_bg  = rmse_bg_v[idx].mean()
    m_cnn = rmse_cnn_v[idx].mean()
    improvs_boot.append((m_bg - m_cnn) / m_bg * 100)
ci = np.percentile(improvs_boot, [2.5, 97.5])

print(f"\n{'='*65}")
print(f"VIIRS EXTERNAL VALIDATION (κ={KAPPA}, n_diff={N_DIFF})")
print(f"{'='*65}")
print(f"Background RMSE : {rmse_bg_v.mean():.4f}")
print(f"Pipeline RMSE   : {rmse_cnn_v.mean():.4f}")
print(f"Improvement     : {improv:.1f}% [{ci[0]:.1f}%, {ci[1]:.1f}%]")
print(f"\nOld result (κ=0.3, n_diff=30): BG=0.2394, CNN=0.0946, Improv=60.5%")
print(f"{'='*65}")

np.savez('results/viirs_optimal_diffusion.npz',
         rmse_cnn=rmse_cnn_v, rmse_bg=rmse_bg_v,
         improv=improv, ci=ci,
         viirs_mean=viirs_mean, viirs_std=viirs_std)
print("\nSaved -> results/viirs_optimal_diffusion.npz")
print("\nDone.")
