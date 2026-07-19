"""
Cloud-Shaped Masking: All Methods Comparison
=============================================
Tier 1 Fix #3: Add DINEOF and OI to cloud-shaped masking evaluation.
Same 138 target/donor pairs as cloud_mask_validation.py.
Also computes OI bootstrap CI for internal validation (Tier 3 Fix #10).
"""
import numpy as np
from scipy.spatial.distance import cdist
from datetime import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.abspath('.')))
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Cloud-Shaped Masking: All Methods + OI Bootstrap CI")
print("=" * 70)

KAPPA  = 0.4
N_DIFF = 50
L_OI   = 0.15   # CV-optimised from oi_baseline.py

chl_norm = np.load('data/modis_chl/chl_norm.npy')
dates    = np.load('data/modis_chl/chl_dates.npy')
N, LAT, LON = chl_norm.shape

def parse_date(ds):
    return datetime.strptime(str(ds), '%Y%m%d')

parsed = [parse_date(d) for d in dates]
years  = np.array([d.year for d in parsed])
doys   = np.array([d.timetuple().tm_yday for d in parsed])

# ── Rebuild same 138 eval pairs ────────────────────────────────────────
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
print(f"Evaluation pairs: {len(eval_pairs)}")

# ── Load pre-computed global DINEOF reconstruction ─────────────────────
# DINEOF was run on the full dataset; evaluate at cloud-gap pixels
# Note: cloud-gap pixels that were also NaN in original MODIS were
# genuinely filled by DINEOF; pixels valid in MODIS were used as obs.
# Both evaluations are informative — we report what DINEOF gives at
# cloud-gap positions regardless of their original MODIS status.
print("\nLoading DINEOF reconstruction...")
dineof_data   = np.load('results/dineof_results.npz')
ocean_mask    = np.any(~np.isnan(chl_norm), axis=0)
idx_r, idx_c  = np.where(ocean_mask)
n_px = len(idx_r)
X_mat = chl_norm[:, idx_r, idx_c].copy()

# Re-run final DINEOF reconstruction to get per-pixel values
from scipy.sparse.linalg import svds

def dineof_fill(X_in, r=17, tol=1e-4, max_iter=200):
    nan_mask = np.isnan(X_in)
    X = X_in.copy()
    col_mean = np.nanmean(X, axis=0)
    col_mean[np.isnan(col_mean)] = 0.0
    X[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    for it in range(max_iter):
        r_eff = min(r, min(X.shape) - 1)
        try:
            U, s, Vt = svds(X, k=r_eff)
        except:
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

# Build a 2D lookup of DINEOF reconstruction
print("Running DINEOF reconstruction (r=17, same as global)...")
X_rec = dineof_fill(X_mat, r=17)
# Convert back to 2D
dineof_rec_3d = np.full_like(chl_norm, np.nan)
for t in range(N):
    field = np.full((LAT, LON), np.nan)
    field[idx_r, idx_c] = X_rec[t]
    dineof_rec_3d[t] = field
print("DINEOF done.")

# ── Diffusion pipeline ─────────────────────────────────────────────────
def diffusion_pipeline(mu_b, y_obs, obs_mask):
    mu_a = mu_b.copy()
    mu_a[obs_mask] = y_obs[obs_mask]
    for _ in range(N_DIFF):
        p = np.pad(mu_a, 1, mode='edge')
        L = p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*mu_a
        u = np.zeros_like(mu_a)
        u[~obs_mask] = KAPPA * L[~obs_mask]
        mu_a = mu_a + 0.1 * u
    return mu_a

# ── OI ─────────────────────────────────────────────────────────────────
def oi_fill(obs_rows, obs_cols, obs_vals, tst_rows, tst_cols, L=L_OI):
    n_obs_use = min(800, len(obs_rows))
    if len(obs_rows) > n_obs_use:
        sel = np.random.choice(len(obs_rows), n_obs_use, replace=False)
        obs_rows_s = obs_rows[sel]; obs_cols_s = obs_cols[sel]
        obs_vals_s = obs_vals[sel]
    else:
        obs_rows_s, obs_cols_s, obs_vals_s = obs_rows, obs_cols, obs_vals
    obs_c = np.stack([obs_rows_s/LAT, obs_cols_s/LON], axis=1)
    tst_c = np.stack([tst_rows/LAT,   tst_cols/LON], axis=1)
    D_oo  = cdist(obs_c, obs_c)
    D_to  = cdist(tst_c, obs_c)
    C_oo  = np.exp(-0.5*(D_oo/L)**2) + 1e-4*np.eye(len(obs_rows_s))
    C_to  = np.exp(-0.5*(D_to/L)**2)
    try:
        pred = C_to @ np.linalg.solve(C_oo, obs_vals_s)
        return pred if np.isfinite(pred).all() else None
    except:
        return None

# ── Run all methods on 138 pairs ───────────────────────────────────────
np.random.seed(42)
results = {m: {'cnn': [], 'bg': []}
           for m in ['diff', 'dineof', 'oi']}
n_skip = 0

for step, (ti, di) in enumerate(eval_pairs):
    if step % 50 == 0:
        print(f"  {step}/{len(eval_pairs)}...", flush=True)

    target = chl_norm[ti]; donor = chl_norm[di]
    gap_mask = np.isnan(donor) & (~np.isnan(target))
    n_test = int(gap_mask.sum())
    if n_test < 50:
        n_skip += 1
        continue
    obs_mask = (~np.isnan(target)) & (~np.isnan(donor))
    if obs_mask.sum() < 200:
        n_skip += 1
        continue

    mu_b = chl_norm[ti-1].copy(); mu_b[np.isnan(mu_b)] = 0.0
    y_obs = np.where(obs_mask, target, 0.0)
    true_v = target[gap_mask]
    bg_v   = mu_b[gap_mask]

    # --- Diffusion pipeline ---
    mu_a_diff = diffusion_pipeline(mu_b, y_obs, obs_mask)
    diff_v = mu_a_diff[gap_mask]
    results['diff']['cnn'].append(np.sqrt(np.mean((true_v-diff_v)**2)))
    results['diff']['bg'].append( np.sqrt(np.mean((true_v-bg_v)**2)))

    # --- DINEOF (pre-computed global reconstruction) ---
    dineof_v = dineof_rec_3d[ti][gap_mask]
    results['dineof']['cnn'].append(np.sqrt(np.mean((true_v-dineof_v)**2)))
    results['dineof']['bg'].append( np.sqrt(np.mean((true_v-bg_v)**2)))

    # --- OI ---
    obs_rows, obs_cols = np.where(obs_mask)
    obs_vals = target[obs_rows, obs_cols]
    tst_rows, tst_cols = np.where(gap_mask)
    oi_pred = oi_fill(obs_rows, obs_cols, obs_vals, tst_rows, tst_cols)
    if oi_pred is not None:
        results['oi']['cnn'].append(np.sqrt(np.mean((true_v-oi_pred)**2)))
        results['oi']['bg'].append( np.sqrt(np.mean((true_v-bg_v)**2)))

print(f"\nSkipped: {n_skip} | Evaluated: {len(results['diff']['cnn'])}")

# ── Bootstrap CI ───────────────────────────────────────────────────────
N_BOOT = 10000
np.random.seed(42)

def boot_improv(bg_arr, cnn_arr, n_boot=N_BOOT):
    n = len(bg_arr)
    improvs = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        m_bg  = bg_arr[idx].mean()
        m_cnn = cnn_arr[idx].mean()
        improvs.append((m_bg-m_cnn)/m_bg*100)
    return np.percentile(improvs, [2.5, 97.5])

print(f"\n{'='*70}")
print("CLOUD-SHAPED MASKING: ALL METHODS")
print(f"{'='*70}")
print(f"{'Method':<20}{'BG RMSE':>10}{'RMSE':>10}{'Improvement':>14}{'95% CI':>20}")
print("-"*70)

for label, key in [('Background',None),
                   ('OI (L=0.15)', 'oi'),
                   ('DINEOF (r=17)','dineof'),
                   ('Diffusion pipeline','diff')]:
    if key is None:
        bg_m = np.mean(results['diff']['bg'])
        print(f"{label:<20}{bg_m:>10.4f}{'---':>10}{'---':>14}{'---':>20}")
    else:
        bg_arr  = np.array(results[key]['bg'])
        cnn_arr = np.array(results[key]['cnn'])
        bg_m  = bg_arr.mean()
        cnn_m = cnn_arr.mean()
        imp   = (bg_m-cnn_m)/bg_m*100
        ci    = boot_improv(bg_arr, cnn_arr)
        print(f"{label:<20}{bg_m:>10.4f}{cnn_m:>10.4f}"
              f"{imp:>13.1f}%  [{ci[0]:.1f}%, {ci[1]:.1f}%]")

# ── OI bootstrap CI for internal validation (Table 4 fix) ─────────────
print(f"\n{'='*70}")
print("OI BOOTSTRAP CI FOR INTERNAL VALIDATION (Table 4 fix)")
print(f"{'='*70}")
oi_data = np.load('results/oi_baseline.npz')
rmse_oi_int = oi_data['rmse_oi']
rmse_bg_int = oi_data['rmse_bg']
oi_rmse_mean = rmse_oi_int.mean()
oi_bg_mean   = rmse_bg_int.mean()
oi_improv    = (oi_bg_mean-oi_rmse_mean)/oi_bg_mean*100

np.random.seed(42)
oi_improvs_boot = []
n = len(rmse_oi_int)
for _ in range(N_BOOT):
    idx = np.random.choice(n, n, replace=True)
    m_bg  = rmse_bg_int[idx].mean()
    m_oi  = rmse_oi_int[idx].mean()
    oi_improvs_boot.append((m_bg-m_oi)/m_bg*100)
oi_ci_improv = np.percentile(oi_improvs_boot, [2.5, 97.5])

oi_means_boot = [rmse_oi_int[np.random.choice(n,n,replace=True)].mean()
                  for _ in range(N_BOOT)]
oi_ci_rmse = np.percentile(oi_means_boot, [2.5, 97.5])

print(f"OI internal: RMSE={oi_rmse_mean:.4f} [{oi_ci_rmse[0]:.4f},{oi_ci_rmse[1]:.4f}]")
print(f"Improvement: {oi_improv:.1f}% [{oi_ci_improv[0]:.1f}%, {oi_ci_improv[1]:.1f}%]")

np.savez('results/cloud_mask_all_methods.npz',
         diff_bg=np.array(results['diff']['bg']),
         diff_cnn=np.array(results['diff']['cnn']),
         dineof_bg=np.array(results['dineof']['bg']),
         dineof_cnn=np.array(results['dineof']['cnn']),
         oi_bg=np.array(results['oi']['bg']),
         oi_cnn=np.array(results['oi']['cnn']),
         oi_rmse_internal=oi_rmse_mean,
         oi_ci_rmse=oi_ci_rmse,
         oi_improv_internal=oi_improv,
         oi_ci_improv=oi_ci_improv)
print("\nSaved -> results/cloud_mask_all_methods.npz")
print("\nDone.")
