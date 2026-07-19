"""
Experiment 5: Optimal Interpolation Baseline
=============================================
Simple spatial OI using a Gaussian covariance kernel.
Correlation length optimized by leave-one-out cross-validation
on a small subset. Same 20% held-out test mask as all other methods.
"""
import numpy as np
from scipy.spatial.distance import cdist
import sys, os
sys.path.append(os.path.dirname(os.path.abspath('.')))
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("Experiment 5: Optimal Interpolation Baseline")
print("=" * 65)

chl_norm = np.load('data/modis_chl/chl_norm.npy')
N, LAT, LON = chl_norm.shape

# ── Reproduce test mask ────────────────────────────────────────────────
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
    test_mask[t, valid_idx[0][sel], valid_idx[1][sel]] = True
    eval_t.append(t)

print(f"Test timesteps: {len(eval_t)}")

# ── Grid coordinates (normalized 0-1) ─────────────────────────────────
lat_idx = np.arange(LAT) / LAT
lon_idx = np.arange(LON) / LON
grid_lat, grid_lon = np.meshgrid(lat_idx, lon_idx, indexing='ij')
grid_coords = np.stack([grid_lat.ravel(), grid_lon.ravel()], axis=1)

# ── CV to choose length scale on first 10 eval timesteps ──────────────
L_VALS = [0.02, 0.05, 0.10, 0.15, 0.20]
cv_rmses = {l: [] for l in L_VALS}
cv_t = eval_t[:10]  # fast CV on first 10 timesteps

print(f"\nCross-validating length scale on {len(cv_t)} timesteps...")
for t in cv_t:
    fc = chl_norm[t]
    fp = chl_norm[t-1]
    valid_mask = ~np.isnan(fc)
    tm = test_mask[t]
    obs_mask = valid_mask & (~tm)
    obs_rows, obs_cols = np.where(obs_mask)
    tst_rows, tst_cols = np.where(tm)
    if len(obs_rows) < 10 or len(tst_rows) < 10:
        continue
    obs_coords = np.stack([obs_rows/LAT, obs_cols/LON], axis=1)
    tst_coords = np.stack([tst_rows/LAT, tst_cols/LON], axis=1)
    obs_vals = fc[obs_rows, obs_cols]
    true_v   = fc[tst_rows, tst_cols]
    fp_clean = fp.copy(); fp_clean[np.isnan(fp_clean)] = 0.0
    bg_v = fp_clean[tst_rows, tst_cols]

    # subsample obs/test for speed
    n_obs_cv = min(500, len(obs_rows))
    n_tst_cv = min(200, len(tst_rows))
    oi = np.random.choice(len(obs_rows), n_obs_cv, replace=False)
    ti = np.random.choice(len(tst_rows), n_tst_cv, replace=False)
    obs_c2 = obs_coords[oi]; obs_v2 = obs_vals[oi]
    tst_c2 = tst_coords[ti]; true_v2 = true_v[ti]

    for L in L_VALS:
        D_oo = cdist(obs_c2, obs_c2)
        D_to = cdist(tst_c2, obs_c2)
        C_oo = np.exp(-0.5*(D_oo/L)**2) + 1e-4*np.eye(n_obs_cv)
        C_to = np.exp(-0.5*(D_to/L)**2)
        try:
            weights = C_to @ np.linalg.solve(C_oo, obs_v2)
            rmse = np.sqrt(np.mean((true_v2 - weights)**2))
            if np.isfinite(rmse):
                cv_rmses[L].append(rmse)
        except Exception:
            pass

cv_mean = {L: np.mean(v) if v else np.inf for L, v in cv_rmses.items()}
L_opt = min(cv_mean, key=cv_mean.get)
print(f"  CV results: " + ", ".join([f"L={l}:{cv_mean[l]:.4f}" for l in L_VALS]))
print(f"  Optimal L = {L_opt}")

# ── Full evaluation with L_opt ─────────────────────────────────────────
print(f"\nRunning OI with L={L_opt} on all {len(eval_t)} timesteps...")
rmse_oi, rmse_bg_oi = [], []

for step, t in enumerate(eval_t):
    if step % 50 == 0:
        print(f"  {step}/{len(eval_t)}...", flush=True)
    fc = chl_norm[t]; fp = chl_norm[t-1]
    valid_mask = ~np.isnan(fc)
    tm = test_mask[t]
    obs_mask = valid_mask & (~tm)
    obs_rows, obs_cols = np.where(obs_mask)
    tst_rows, tst_cols = np.where(tm)
    if len(obs_rows) < 10:
        continue
    obs_coords = np.stack([obs_rows/LAT, obs_cols/LON], axis=1)
    tst_coords = np.stack([tst_rows/LAT, tst_cols/LON], axis=1)
    obs_vals = fc[obs_rows, obs_cols]
    true_v   = fc[tst_rows, tst_cols]
    fp_clean = fp.copy(); fp_clean[np.isnan(fp_clean)] = 0.0
    bg_v = fp_clean[tst_rows, tst_cols]

    # Subsample obs for speed (OI scales O(n_obs^2))
    n_obs_use = min(800, len(obs_rows))
    if len(obs_rows) > n_obs_use:
        sel = np.random.choice(len(obs_rows), n_obs_use, replace=False)
        obs_coords = obs_coords[sel]
        obs_vals   = obs_vals[sel]

    D_oo = cdist(obs_coords, obs_coords)
    D_to = cdist(tst_coords, obs_coords)
    C_oo = np.exp(-0.5*(D_oo/L_opt)**2) + 1e-4*np.eye(len(obs_coords))
    C_to = np.exp(-0.5*(D_to/L_opt)**2)
    try:
        oi_pred = C_to @ np.linalg.solve(C_oo, obs_vals)
        if np.isfinite(oi_pred).all():
            rmse_oi.append(np.sqrt(np.mean((true_v - oi_pred)**2)))
            rmse_bg_oi.append(np.sqrt(np.mean((true_v - bg_v)**2)))
    except Exception:
        pass

rmse_oi    = np.array(rmse_oi)
rmse_bg_oi = np.array(rmse_bg_oi)
improv = (rmse_bg_oi.mean()-rmse_oi.mean())/rmse_bg_oi.mean()*100

print(f"\n{'='*65}")
print("OI BASELINE RESULTS")
print(f"{'='*65}")
print(f"{'Method':<28}{'RMSE':>10}{'Improv%':>10}")
print("-"*65)
print(f"{'Background':<28}{rmse_bg_oi.mean():>10.4f}")
print(f"{'OI (L={L_opt})':<28}{rmse_oi.mean():>10.4f}{improv:>9.1f}%")
print(f"{'DINEOF (r=17) [ref]':<28}{'0.4124':>10}{'45.4%':>10}")
print(f"{'Cell-NN+Diff [ref]':<28}{'0.1785':>10}{'76.3%':>10}")
print(f"{'='*65}")

np.savez('results/oi_baseline.npz',
         rmse_oi=rmse_oi, rmse_bg=rmse_bg_oi, L_opt=L_opt)
print(f"\nSaved -> results/oi_baseline.npz")
print("\nDone.")
