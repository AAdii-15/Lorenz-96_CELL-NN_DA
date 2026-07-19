"""
Chl-a Reconstruction with Optimised Diffusion (κ=0.4, n_diff=50)
==================================================================
Re-runs the main internal validation, external VIIRS validation,
cloud-shaped masking, and bootstrap CIs using the
diffusion-sensitivity-optimised parameters κ=0.4, n_diff=50.
Same test mask (seed=42), same protocols as all prior runs.
All previous numbers used κ=0.3, n_diff=30 without justification.
These are now justified by the sensitivity analysis in Exp A.
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath('.')))
from src.da_cellnn import cellnn_da_step
import xarray as xr, glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Chl-a Reconstruction: Optimised Diffusion (κ=0.4, n_diff=50)")
print("=" * 70)

KAPPA  = 0.4
N_DIFF = 50

chl_norm = np.load('data/modis_chl/chl_norm.npy')
chl_mean = np.load('data/modis_chl/chl_mean.npy')[0]
chl_std  = np.load('data/modis_chl/chl_std.npy')[0]
dates    = np.load('data/modis_chl/chl_dates.npy')
N, LAT, LON = chl_norm.shape
date_strs = [str(d) for d in dates]

# ── Reproduce identical test mask ─────────────────────────────────────
np.random.seed(42)
test_mask = np.zeros((N, LAT, LON), dtype=bool)
eval_t = []
for t in range(1, N):
    fc = chl_norm[t]
    valid_mask = ~np.isnan(fc)
    if valid_mask.sum() < 500:
        continue
    vi = np.where(valid_mask)
    n_valid = len(vi[0])
    sel = np.random.choice(n_valid, int(0.2*n_valid), replace=False)
    test_mask[t, vi[0][sel], vi[1][sel]] = True
    eval_t.append(t)
print(f"Test timesteps: {len(eval_t)}")

def diffusion_pipeline(mu_b_2d, y_obs_2d, obs_mask_2d,
                       kappa=KAPPA, n_diff=N_DIFF):
    """Diffusion-only pipeline (attribution ablation confirmed
    Cell-NN adds zero; diffusion is the sole skill driver)."""
    mu_a = mu_b_2d.copy()
    mu_a[obs_mask_2d] = y_obs_2d[obs_mask_2d]
    for _ in range(n_diff):
        p = np.pad(mu_a, 1, mode='edge')
        L = p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*mu_a
        u = np.zeros_like(mu_a)
        u[~obs_mask_2d] = kappa * L[~obs_mask_2d]
        mu_a = mu_a + 0.1 * u
    return mu_a

# ── PART 1: Internal validation ───────────────────────────────────────
print("\nPart 1: Internal validation (random 20% mask)...")
rmse_cnn_new, rmse_bg_new = [], []

for t in eval_t:
    fc = chl_norm[t]; fp = chl_norm[t-1]
    valid_mask = ~np.isnan(fc)
    tm = test_mask[t]
    mu_b = fp.copy(); mu_b[np.isnan(mu_b)] = 0.0
    obs_mask = valid_mask & (~tm)
    y_obs = np.where(obs_mask, fc, 0.0)
    mu_a = diffusion_pipeline(mu_b, y_obs, obs_mask)
    rows, cols = np.where(tm)
    true_v = fc[rows, cols]
    rmse_cnn_new.append(np.sqrt(np.mean((true_v - mu_a[rows,cols])**2)))
    rmse_bg_new.append(np.sqrt(np.mean((true_v - mu_b[rows,cols])**2)))

rmse_cnn_new = np.array(rmse_cnn_new)
rmse_bg_new  = np.array(rmse_bg_new)
improv_int = (rmse_bg_new.mean()-rmse_cnn_new.mean())/rmse_bg_new.mean()*100
print(f"  BG={rmse_bg_new.mean():.4f}, Pipeline={rmse_cnn_new.mean():.4f}, "
      f"Improv={improv_int:.1f}%")

# ── PART 2: External VIIRS validation ────────────────────────────────
print("\nPart 2: External VIIRS validation...")
chl_reconstructed = np.load('data/modis_chl/chl_reconstructed.npy')
VIIRS_DIR = "data/viirs_chl/nc_files"
viirs_files = sorted(glob.glob(f"{VIIRS_DIR}/*.nc"))
viirs_by_date = {}
for f in viirs_files:
    fname = os.path.basename(f)
    fdate = fname.split('.')[1].split('_')[0]
    ds = xr.open_dataset(f)
    v = np.log10(ds['chlor_a'].values)
    v[~np.isfinite(v)] = np.nan
    viirs_mean = np.load('data/modis_chl/chl_mean.npy')[0]
    viirs_std  = np.load('data/modis_chl/chl_std.npy')[0]
    viirs_by_date[fdate] = (v - viirs_mean) / viirs_std
    ds.close()

# Re-run reconstruction with new kappa/n_diff to get updated field
print("  Re-building reconstruction with κ=0.4, n_diff=50...")
chl_reconstructed_new = np.zeros_like(chl_norm)
chl_reconstructed_new[0] = chl_norm[0].copy()
chl_reconstructed_new[0][np.isnan(chl_reconstructed_new[0])] = 0.0

for t in range(1, N):
    fc = chl_norm[t]; fp = chl_reconstructed_new[t-1]
    valid_mask = ~np.isnan(fc)
    mu_b = fp.copy()
    obs_mask = valid_mask
    y_obs = np.where(obs_mask, fc, 0.0)
    chl_reconstructed_new[t] = diffusion_pipeline(mu_b, y_obs, obs_mask)

rmse_viirs_cnn, rmse_viirs_bg = [], []
common_dates = sorted(set(date_strs) & set(viirs_by_date.keys()))
for d in common_dates:
    ti = date_strs.index(d)
    cnn_field = chl_reconstructed_new[ti]
    bg_field  = chl_norm[ti]
    viirs_field = viirs_by_date[d]
    vm = ~np.isnan(viirs_field)
    if vm.sum() == 0:
        continue
    bg_v = bg_field[vm]
    bg_v[np.isnan(bg_v)] = 0.0
    rmse_viirs_cnn.append(np.sqrt(np.mean((viirs_field[vm]-cnn_field[vm])**2)))
    rmse_viirs_bg.append(np.sqrt(np.mean((viirs_field[vm]-bg_v)**2)))

rmse_viirs_cnn = np.array(rmse_viirs_cnn)
rmse_viirs_bg  = np.array(rmse_viirs_bg)
improv_ext = (rmse_viirs_bg.mean()-rmse_viirs_cnn.mean())/rmse_viirs_bg.mean()*100
print(f"  BG={rmse_viirs_bg.mean():.4f}, Pipeline={rmse_viirs_cnn.mean():.4f}, "
      f"Improv={improv_ext:.1f}%")

# ── PART 3: Cloud masking with new params ─────────────────────────────
print("\nPart 3: Cloud-shaped masking (new params)...")
def parse_date(ds):
    return datetime.strptime(str(ds), '%Y%m%d')
parsed = [parse_date(d) for d in dates]
years  = np.array([d.year for d in parsed])
doys   = np.array([d.timetuple().tm_yday for d in parsed])

eval_pairs = []
for ti in range(1, N):
    if years[ti] not in [2017, 2018, 2019]:
        continue
    cands = [di for di in range(N)
             if years[di] in [2015, 2016]
             and abs(doys[di]-doys[ti]) <= 8]
    if not cands:
        continue
    best = min(cands, key=lambda di: abs(doys[di]-doys[ti]))
    eval_pairs.append((ti, best))

rmse_cloud_cnn, rmse_cloud_bg = [], []
for ti, di in eval_pairs:
    target = chl_norm[ti]; donor = chl_norm[di]
    gap_mask = np.isnan(donor) & (~np.isnan(target))
    if gap_mask.sum() < 50:
        continue
    obs_mask = (~np.isnan(target)) & (~np.isnan(donor))
    if obs_mask.sum() < 200:
        continue
    mu_b = chl_norm[ti-1].copy(); mu_b[np.isnan(mu_b)] = 0.0
    y_obs = np.where(obs_mask, target, 0.0)
    mu_a = diffusion_pipeline(mu_b, y_obs, obs_mask)
    true_v = target[gap_mask]
    rmse_cloud_cnn.append(np.sqrt(np.mean((true_v-mu_a[gap_mask])**2)))
    rmse_cloud_bg.append(np.sqrt(np.mean((true_v-mu_b[gap_mask])**2)))

rmse_cloud_cnn = np.array(rmse_cloud_cnn)
rmse_cloud_bg  = np.array(rmse_cloud_bg)
improv_cloud = (rmse_cloud_bg.mean()-rmse_cloud_cnn.mean())/rmse_cloud_bg.mean()*100
print(f"  BG={rmse_cloud_bg.mean():.4f}, Pipeline={rmse_cloud_cnn.mean():.4f}, "
      f"Improv={improv_cloud:.1f}%")

# ── PART 4: Bootstrap CIs ─────────────────────────────────────────────
print("\nPart 4: Bootstrap CIs (n=10,000)...")
np.random.seed(42)
N_BOOT = 10000

def boot_ci(arr_a, arr_b, n_boot=N_BOOT):
    n = len(arr_a)
    improvs = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        m_a = arr_a[idx].mean(); m_b = arr_b[idx].mean()
        improvs.append((m_a-m_b)/m_a*100)
    return np.percentile(improvs, [2.5, 97.5])

ci_int   = boot_ci(rmse_bg_new,     rmse_cnn_new)
ci_ext   = boot_ci(rmse_viirs_bg,   rmse_viirs_cnn)
ci_cloud = boot_ci(rmse_cloud_bg,   rmse_cloud_cnn)

# ── FINAL SUMMARY ─────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"OPTIMISED DIFFUSION RESULTS (κ={KAPPA}, n_diff={N_DIFF})")
print(f"{'='*70}")
print(f"\nInternal validation (random 20% mask):")
print(f"  BG={rmse_bg_new.mean():.4f}, Pipeline={rmse_cnn_new.mean():.4f}, "
      f"Improv={improv_int:.1f}% [{ci_int[0]:.1f}%, {ci_int[1]:.1f}%]")
print(f"\nExternal VIIRS validation:")
print(f"  BG={rmse_viirs_bg.mean():.4f}, Pipeline={rmse_viirs_cnn.mean():.4f}, "
      f"Improv={improv_ext:.1f}% [{ci_ext[0]:.1f}%, {ci_ext[1]:.1f}%]")
print(f"\nCloud-shaped masking:")
print(f"  BG={rmse_cloud_bg.mean():.4f}, Pipeline={rmse_cloud_cnn.mean():.4f}, "
      f"Improv={improv_cloud:.1f}% [{ci_cloud[0]:.1f}%, {ci_cloud[1]:.1f}%]")
print(f"\nComparison with OLD params (κ=0.3, n_diff=30):")
print(f"  Internal:  0.1785 → {rmse_cnn_new.mean():.4f} "
      f"({(0.1785-rmse_cnn_new.mean())/0.1785*100:.1f}% better)")
print(f"  Cloud:     0.7558 → {rmse_cloud_cnn.mean():.4f}")
print(f"{'='*70}")

np.savez('results/chl_optimal_diffusion.npz',
         rmse_cnn_internal=rmse_cnn_new, rmse_bg_internal=rmse_bg_new,
         rmse_cnn_viirs=rmse_viirs_cnn, rmse_bg_viirs=rmse_viirs_bg,
         rmse_cnn_cloud=rmse_cloud_cnn, rmse_bg_cloud=rmse_cloud_bg,
         ci_int=ci_int, ci_ext=ci_ext, ci_cloud=ci_cloud,
         kappa=KAPPA, n_diff=N_DIFF)
print("\nSaved -> results/chl_optimal_diffusion.npz")
print("\nDone.")
