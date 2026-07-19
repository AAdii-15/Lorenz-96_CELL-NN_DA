"""
Multi-method comparison figure -- Stage 2: OI and DINEOF
=============================================================
Reconstructs REAL cloud gaps (not synthetic held-out pixels) for the
4 selected dates. Reuses the already-established optimal
hyperparameters from Table 6 (OI: L=0.15, DINEOF: r=17) -- no
re-running of cross-validation.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings('ignore')

print("="*65)
print("Multi-method figure -- Stage 2: OI and DINEOF (real gaps)")
print("="*65)

chl_norm = np.load('data/modis_chl/chl_norm.npy')
dates = np.load('data/modis_chl/chl_dates.npy')
N, LAT, LON = chl_norm.shape

date_strs = [str(d) for d in dates]
target_dates = ['20190829', '20190914', '20170501', '20170125']
target_idx = [date_strs.index(d) for d in target_dates]

ocean_mask = np.any(~np.isnan(chl_norm), axis=0)

# ── OI: per-date, real gaps, L=0.15 (already established optimal) ──
L_OPT = 0.15
print(f"\nRunning OI (L={L_OPT}) on real cloud gaps for 4 dates...")
oi_results = {}

for d, idx in zip(target_dates, target_idx):
    field_curr = chl_norm[idx]
    valid_mask = ~np.isnan(field_curr)
    gap_mask = ocean_mask & ~valid_mask

    obs_rows, obs_cols = np.where(valid_mask)
    tst_rows, tst_cols = np.where(gap_mask)
    print(f"  {d}: {len(obs_rows)} obs, {len(tst_rows)} real gap pixels to fill")

    obs_coords = np.stack([obs_rows/LAT, obs_cols/LON], axis=1)
    tst_coords = np.stack([tst_rows/LAT, tst_cols/LON], axis=1)
    obs_vals = field_curr[obs_rows, obs_cols]

    n_obs_use = min(800, len(obs_rows))
    if len(obs_rows) > n_obs_use:
        sel = np.random.RandomState(42).choice(len(obs_rows), n_obs_use, replace=False)
        obs_coords_sub = obs_coords[sel]
        obs_vals_sub = obs_vals[sel]
    else:
        obs_coords_sub = obs_coords
        obs_vals_sub = obs_vals

    D_oo = cdist(obs_coords_sub, obs_coords_sub)
    D_to = cdist(tst_coords, obs_coords_sub)
    C_oo = np.exp(-0.5*(D_oo/L_OPT)**2) + 1e-1*np.eye(len(obs_coords_sub))
    C_to = np.exp(-0.5*(D_to/L_OPT)**2)
    oi_pred = C_to @ np.linalg.solve(C_oo, obs_vals_sub)

    oi_field = field_curr.copy()
    oi_field[tst_rows, tst_cols] = oi_pred
    oi_results[d] = oi_field
    print(f"    Done. Filled {len(tst_rows)} pixels.")

np.savez('results/multimethod_oi.npz', target_dates=target_dates,
         **{f'oi_{d}': oi_results[d] for d in target_dates})
print("Saved -> results/multimethod_oi.npz")

# ── DINEOF: whole-matrix run, real gaps only, r=17 ──
print(f"\nBuilding DINEOF working matrix (real gaps only, no held-out masking)...")
idx_r, idx_c = np.where(ocean_mask)
X_mat = chl_norm[:, idx_r, idx_c].copy()
print(f"Matrix shape: {X_mat.shape}")

def dineof_fill(X_in, r, tol=1e-4, max_iter=200):
    nan_mask = np.isnan(X_in)
    X = X_in.copy()
    col_mean = np.nanmean(X, axis=0)
    col_mean[np.isnan(col_mean)] = 0.0
    X[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])

    for it in range(max_iter):
        r_eff = min(r, min(X.shape) - 1)
        U, s, Vt = svds(X, k=r_eff)
        order = np.argsort(s)[::-1]
        U, s, Vt = U[:, order], s[order], Vt[order, :]
        X_low = (U * s) @ Vt
        X_new = X.copy()
        X_new[nan_mask] = X_low[nan_mask]
        change = np.linalg.norm(X_new - X) / max(np.linalg.norm(X), 1e-10)
        X = X_new
        if it % 20 == 0:
            print(f"    iter {it}: change={change:.6f}")
        if change < tol:
            print(f"    converged at iter {it}")
            break
    return X

print(f"\nRunning DINEOF (r=17, tol=1e-4, max_iter=200) on real gaps...")
X_rec = dineof_fill(X_mat, r=17, tol=1e-4, max_iter=200)
print("DINEOF reconstruction complete.")

dineof_results = {}
for d, idx in zip(target_dates, target_idx):
    field = np.full((LAT, LON), np.nan)
    field[idx_r, idx_c] = X_rec[idx]
    dineof_results[d] = field

np.savez('results/multimethod_dineof.npz', target_dates=target_dates,
         **{f'dineof_{d}': dineof_results[d] for d in target_dates})
print("Saved -> results/multimethod_dineof.npz")
print("\nStage 2 done.")
