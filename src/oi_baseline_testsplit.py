"""
OI Baseline -- TEST SPLIT (leakage fix)
=========================================
L is now chosen by CV against val_mask (not test_mask). Final RMSE
is reported only against test_mask.
"""
import numpy as np
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

chl_norm = np.load('data/modis_chl/chl_norm.npy')
N, LAT, LON = chl_norm.shape

split = np.load('results/holdout_split_masks.npz')
held_out_20 = split['held_out_20']
val_mask    = split['val_mask']
test_mask   = split['test_mask']
eval_t      = split['eval_t']

print(f"Test timesteps: {len(eval_t)}")

L_VALS = [0.02, 0.05, 0.10, 0.15, 0.20]
cv_rmses = {l: [] for l in L_VALS}
cv_t = eval_t[:10]

np.random.seed(42)
print(f"\nCV length scale on {len(cv_t)} timesteps (VALIDATION pixels)...")
for t in cv_t:
    fc = chl_norm[t]
    valid_mask = ~np.isnan(fc)
    obs_mask = valid_mask & (~held_out_20[t])
    vm = val_mask[t]
    obs_rows, obs_cols = np.where(obs_mask)
    val_rows, val_cols = np.where(vm)
    if len(obs_rows) < 10 or len(val_rows) < 10:
        continue
    obs_coords = np.stack([obs_rows/LAT, obs_cols/LON], axis=1)
    val_coords = np.stack([val_rows/LAT, val_cols/LON], axis=1)
    obs_vals = fc[obs_rows, obs_cols]
    true_v   = fc[val_rows, val_cols]
    n_obs_cv = min(500, len(obs_rows))
    n_val_cv = min(200, len(val_rows))
    oi = np.random.choice(len(obs_rows), n_obs_cv, replace=False)
    vi = np.random.choice(len(val_rows), n_val_cv, replace=False)
    obs_c2 = obs_coords[oi]; obs_v2 = obs_vals[oi]
    val_c2 = val_coords[vi]; true_v2 = true_v[vi]
    for L in L_VALS:
        D_oo = cdist(obs_c2, obs_c2)
        D_to = cdist(val_c2, obs_c2)
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

print(f"\nFull eval with L={L_opt}, scoring TEST pixels only...")
rmse_oi, rmse_bg_oi = [], []
for step, t in enumerate(eval_t):
    fc = chl_norm[t]; fp = chl_norm[t-1]
    valid_mask = ~np.isnan(fc)
    obs_mask = valid_mask & (~held_out_20[t])
    tm = test_mask[t]
    obs_rows, obs_cols = np.where(obs_mask)
    tst_rows, tst_cols = np.where(tm)
    if len(obs_rows) < 10 or len(tst_rows) == 0:
        continue
    obs_coords = np.stack([obs_rows/LAT, obs_cols/LON], axis=1)
    tst_coords = np.stack([tst_rows/LAT, tst_cols/LON], axis=1)
    obs_vals = fc[obs_rows, obs_cols]
    true_v   = fc[tst_rows, tst_cols]
    n_obs_use = min(800, len(obs_rows))
    if len(obs_rows) > n_obs_use:
        sel = np.random.choice(len(obs_rows), n_obs_use, replace=False)
        obs_coords = obs_coords[sel]
        obs_vals   = obs_vals[sel]
    D_oo = cdist(obs_coords, obs_coords)
    D_to = cdist(tst_coords, obs_coords)
    C_oo = np.exp(-0.5*(D_oo/L_opt)**2) + 1e-4*np.eye(len(obs_coords))
    C_to = np.exp(-0.5*(D_to/L_opt)**2)
    fp_clean = fp.copy(); fp_clean[np.isnan(fp_clean)] = 0.0
    bg_v = fp_clean[tst_rows, tst_cols]
    try:
        oi_pred = C_to @ np.linalg.solve(C_oo, obs_vals)
        if np.isfinite(oi_pred).all():
            rmse_oi.append(np.sqrt(np.mean((true_v - oi_pred)**2)))
            rmse_bg_oi.append(np.sqrt(np.mean((true_v - bg_v)**2)))
    except Exception:
        pass

rmse_oi = np.array(rmse_oi); rmse_bg_oi = np.array(rmse_bg_oi)
improv = (rmse_bg_oi.mean()-rmse_oi.mean())/rmse_bg_oi.mean()*100
print(f"\nBackground RMSE (test-only): {rmse_bg_oi.mean():.4f}")
print(f"OI (L={L_opt}) RMSE (test-only): {rmse_oi.mean():.4f}")
print(f"Improvement: {improv:.1f}%  [was L=0.15, RMSE=0.501, 33.7%]")
np.savez('results/oi_baseline_testsplit.npz', rmse_oi=rmse_oi, rmse_bg=rmse_bg_oi, L_opt=L_opt)
print("Saved -> results/oi_baseline_testsplit.npz")
