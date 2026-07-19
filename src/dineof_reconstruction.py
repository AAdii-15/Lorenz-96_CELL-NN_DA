"""
Experiment 3: DINEOF Reconstruction (Beckers & Rixen 2003)
=========================================================================
Implements DINEOF from scratch and evaluates it on the identical
held-out 20% pixel validation protocol used for Cell-NN and Monte-Carlo,
so all three methods are directly comparable.

Algorithm (standard published DINEOF):
  1. Fill NaNs with field mean
  2. Truncated SVD -> keep r EOFs
  3. Replace only NaN/held-out positions with low-rank reconstruction
  4. Repeat until convergence (||X_new - X_old|| / ||X_old|| < tol)
  5. Cross-validate r on a small subset of observed pixels to pick
     the optimal number of EOFs without overfitting

The matrix operated on is the full (N_timesteps x N_pixels) data matrix,
reshaped from (230, 408, 480). Only ocean pixels (ever valid at least
once) are included -- land pixels are excluded permanently.

Reference: Beckers & Rixen (2003), J. Atmos. Ocean. Technol.
           Alvera-Azcarate et al. (2005), J. Mar. Syst.
"""
import numpy as np
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Experiment 3: DINEOF Reconstruction")
print("=" * 70)

chl_norm = np.load('data/modis_chl/chl_norm.npy')
N, LAT, LON = chl_norm.shape
print(f"Shape: {chl_norm.shape}")

# ── Build spatial mask: pixels with ANY valid obs across all timesteps
ever_valid = np.any(~np.isnan(chl_norm), axis=0)   # (LAT, LON)
n_px = int(ever_valid.sum())
print(f"Ocean pixels (ever valid): {n_px:,} / {LAT*LON:,}")

# ── Build data matrix (N x n_px), NaN where cloud-covered
idx_r, idx_c = np.where(ever_valid)
X_mat = chl_norm[:, idx_r, idx_c].copy()           # (N, n_px)

# ── Reproduce EXACT same test mask as all prior runs ──────────────────
np.random.seed(42)
test_mask_2d = np.zeros((N, LAT, LON), dtype=bool)
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
    test_mask_2d[t, valid_idx[0][sel], valid_idx[1][sel]] = True
    eval_t.append(t)

# Map test_mask to the ocean-pixel matrix
test_mask_mat = test_mask_2d[:, idx_r, idx_c]    # (N, n_px)
print(f"Held-out test pixels: {int(test_mask_mat.sum()):,} "
      f"(identical to Cell-NN/Monte-Carlo runs)")

# ── Build the working matrix: hide test pixels + keep NaN for clouds ──
X_work = X_mat.copy()
X_work[test_mask_mat] = np.nan

# ── DINEOF core ────────────────────────────────────────────────────────
def dineof_fill(X_in, r, tol=1e-4, max_iter=200):
    """
    Standard DINEOF iteration for a fixed number of EOFs r.
    Returns the reconstructed full matrix.
    """
    nan_mask = np.isnan(X_in)
    X = X_in.copy()
    col_mean = np.nanmean(X, axis=0)
    col_mean[np.isnan(col_mean)] = 0.0
    X[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])

    prev_norm = None
    for it in range(max_iter):
        r_eff = min(r, min(X.shape) - 1)
        try:
            U, s, Vt = svds(X, k=r_eff)
        except Exception:
            break
        # sort descending
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

# ── Cross-validate r on a small subset of observed pixels ─────────────
print("\nCross-validating number of EOFs (r = 1 to 20)...")
R_VALS = list(range(1, 21))
CV_FRAC = 0.02   # 2% of observed pixels for CV (fast)

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

# ── Final DINEOF reconstruction with optimal r ─────────────────────────
print(f"\nRunning final DINEOF reconstruction (r={r_opt}, "
      f"tol=1e-4, max_iter=200)...")
X_rec_final = dineof_fill(X_work, r_opt, tol=1e-4, max_iter=200)
print("Done.")

# ── Evaluate on held-out test pixels ──────────────────────────────────
rmse_dineof, rmse_bg = [], []
for t in eval_t:
    tm = test_mask_mat[t]
    if tm.sum() == 0:
        continue
    true_v = X_mat[t, tm]
    pred_v = X_rec_final[t, tm]
    bg_row = t - 1
    bg_v = X_mat[bg_row, tm]
    bg_v[np.isnan(bg_v)] = 0.0

    rmse_dineof.append(np.sqrt(np.mean((true_v - pred_v) ** 2)))
    rmse_bg.append(np.sqrt(np.mean((true_v - bg_v) ** 2)))

rmse_dineof = np.array(rmse_dineof)
rmse_bg = np.array(rmse_bg)
improv = (rmse_bg.mean() - rmse_dineof.mean()) / rmse_bg.mean() * 100

print(f"\n{'='*70}")
print("EXPERIMENT 3 RESULTS -- DINEOF vs Cell-NN vs Monte-Carlo")
print(f"{'='*70}")
print(f"{'Method':<30}{'RMSE':>10}{'Improvement vs BG':>20}")
print("-"*70)
print(f"{'Background (persistence)':<30}{rmse_bg.mean():>10.4f}{'---':>20}")
print(f"{'DINEOF (r=' + str(r_opt) + ' EOFs)':<30}"
      f"{rmse_dineof.mean():>10.4f}{improv:>19.1f}%")
print(f"{'Cell-NN + Diffusion':<30}{0.1785:>10.4f}{'76.3%':>20}")
print(f"{'Monte-Carlo (causal)':<30}{0.1199:>10.4f}{'84.1%':>20}")
print(f"{'='*70}")
print(f"\nDINEOF optimal EOFs: r={r_opt}")
print(f"All methods evaluated on IDENTICAL held-out test pixels (seed=42)")

np.savez('results/dineof_results.npz',
         rmse_dineof=rmse_dineof, rmse_bg=rmse_bg,
         r_opt=r_opt, cv_rmses=np.array(cv_rmses))
print("\nSaved -> results/dineof_results.npz")
print("\nDone.")
