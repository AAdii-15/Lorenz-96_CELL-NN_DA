"""
DINEOF -- TEST SPLIT (leakage fix)
=====================================
r is chosen exactly as before (2% CV subset, seed=99 -- never touched
the held-out 20%, so no change needed there). Only the reconstruction
input mask and final scoring move: the model now hides the FULL
original 20% (held_out_20) during reconstruction (unchanged), but is
scored only against test_mask instead of the full 20%.
"""
import numpy as np
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings('ignore')

chl_norm = np.load('data/modis_chl/chl_norm.npy')
N, LAT, LON = chl_norm.shape

split = np.load('results/holdout_split_masks.npz')
held_out_20 = split['held_out_20']
test_mask   = split['test_mask']
eval_t      = split['eval_t']

ever_valid = np.any(~np.isnan(chl_norm), axis=0)
n_px = int(ever_valid.sum())
idx_r, idx_c = np.where(ever_valid)
X_mat = chl_norm[:, idx_r, idx_c].copy()

# hide the FULL original 20% -- unchanged from before
held_out_mat = held_out_20[:, idx_r, idx_c]
test_mask_mat = test_mask[:, idx_r, idx_c]
print(f"Held-out (hidden from SVD): {int(held_out_mat.sum()):,}")
print(f"Test pixels (scored):       {int(test_mask_mat.sum()):,}")

X_work = X_mat.copy()
X_work[held_out_mat] = np.nan

def dineof_fill(X_in, r, tol=1e-4, max_iter=200):
    nan_mask = np.isnan(X_in)
    X = X_in.copy()
    col_mean = np.nanmean(X, axis=0)
    col_mean[np.isnan(col_mean)] = 0.0
    X[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    for it in range(max_iter):
        r_eff = min(r, min(X.shape) - 1)
        try:
            U, s, Vt = svds(X, k=r_eff)
        except Exception:
            break
        order = np.argsort(s)[::-1]
        U, s, Vt = U[:, order], s[order], Vt[order, :]
        X_low = (U * s) @ Vt
        X_new = X.copy()
        X_new[nan_mask] = X_low[nan_mask]
        change = np.linalg.norm(X_new - X) / max(np.linalg.norm(X), 1e-10)
        X = X_new
        if change < tol:
            break
    return X

print("\nCross-validating r (1-20) -- SAME 2% subset method as original, seed=99...")
R_VALS = list(range(1, 21))
CV_FRAC = 0.02
np.random.seed(99)
obs_mask_mat = ~np.isnan(X_work)
obs_idx_flat = np.where(obs_mask_mat.flatten())[0]
n_cv = int(CV_FRAC * len(obs_idx_flat))
cv_sel = np.random.choice(len(obs_idx_flat), n_cv, replace=False)
cv_flat_idx = obs_idx_flat[cv_sel]
cv_rows = cv_flat_idx // X_work.shape[1]
cv_cols = cv_flat_idx % X_work.shape[1]
X_cv = X_work.copy()
true_cv = X_work[cv_rows, cv_cols].copy()
X_cv[cv_rows, cv_cols] = np.nan
cv_rmses = []
for r in R_VALS:
    X_rec = dineof_fill(X_cv, r, tol=1e-3, max_iter=100)
    rmse_cv = np.sqrt(np.mean((X_rec[cv_rows, cv_cols] - true_cv) ** 2))
    cv_rmses.append(rmse_cv)
    print(f"  r={r:2d}: CV-RMSE={rmse_cv:.4f}")
r_opt = R_VALS[int(np.argmin(cv_rmses))]
print(f"\nOptimal r = {r_opt} (CV-RMSE = {min(cv_rmses):.4f})")

print(f"\nFinal DINEOF reconstruction (r={r_opt})...")
X_rec_final = dineof_fill(X_work, r_opt, tol=1e-4, max_iter=200)

rmse_dineof, rmse_bg = [], []
for t in eval_t:
    tm = test_mask_mat[t]
    if tm.sum() == 0:
        continue
    true_v = X_mat[t, tm]
    pred_v = X_rec_final[t, tm]
    bg_v = X_mat[t-1, tm]
    bg_v = np.where(np.isnan(bg_v), 0.0, bg_v)
    rmse_dineof.append(np.sqrt(np.mean((true_v - pred_v) ** 2)))
    rmse_bg.append(np.sqrt(np.mean((true_v - bg_v) ** 2)))

rmse_dineof = np.array(rmse_dineof); rmse_bg = np.array(rmse_bg)
improv = (rmse_bg.mean() - rmse_dineof.mean()) / rmse_bg.mean() * 100
print(f"\nBackground RMSE (test-only): {rmse_bg.mean():.4f}")
print(f"DINEOF (r={r_opt}) RMSE (test-only): {rmse_dineof.mean():.4f}")
print(f"Improvement: {improv:.1f}%  [was r=17, RMSE=0.412, 45.4%]")
np.savez('results/dineof_testsplit.npz', rmse_dineof=rmse_dineof, rmse_bg=rmse_bg, r_opt=r_opt)
print("Saved -> results/dineof_testsplit.npz")
