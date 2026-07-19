"""
Satellite pipeline cost comparison: Cell-NN pipeline vs DINEOF vs OI
vs Monte Carlo, single-pass wall-clock timing.
=========================================================================
Times only DEPLOYMENT cost (the actual reconstruction run over all 229
timesteps / the full grid), not the one-time hyperparameter search
already done separately for each method (DINEOF r=17, OI L=0.2,
Cell-NN kappa=0.4/n=50) -- same convention Table 5 already uses for
Cell-NN's own timing.
"""
import numpy as np
import time, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.da_cellnn import cellnn_da_step
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cdist
from scipy import stats
from scipy.stats import kstest
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("Satellite Pipeline Cost Comparison (single-pass wall-clock)")
print("=" * 65)

chl_raw  = np.load('data/modis_chl/chl_data.npy')
chl_norm = np.load('data/modis_chl/chl_norm.npy')
chl_mean = np.load('data/modis_chl/chl_mean.npy')
chl_std  = np.load('data/modis_chl/chl_std.npy')
N, LAT, LON = chl_norm.shape

split = np.load('results/holdout_split_masks.npz')
held_out_20 = split['held_out_20']
eval_t = list(split['eval_t'])
print(f"Timesteps: {len(eval_t)}")

results = {}

# ── Cell-NN pipeline (kappa=0.4, n=50) ──────────────────────────────
print("\nTiming Cell-NN pipeline...")
def cellnn_da_2d(mu_b_2d, y_obs_2d, obs_mask_2d, alpha_A=1.0, wb=1.0, kappa=0.4, n_diff=50):
    lat_n, lon_n = mu_b_2d.shape
    mu_b_flat = mu_b_2d.flatten(); y_obs_flat = y_obs_2d.flatten(); obs_flat = obs_mask_2d.flatten()
    mu_a_obs, _ = cellnn_da_step(mu_b_flat[obs_flat], y_obs_flat[obs_flat],
                                  alpha_A=alpha_A, wb=wb, d_tau=5e-2, tau_max=1e-1, eps_conv=1e-6, r_max=50)
    mu_a_flat = mu_b_flat.copy(); mu_a_flat[obs_flat] = mu_a_obs
    mu_a_2d = mu_a_flat.reshape(lat_n, lon_n)
    for _ in range(n_diff):
        mu_pad = np.pad(mu_a_2d, 1, mode='edge')
        laplacian = (mu_pad[:-2,1:-1]+mu_pad[2:,1:-1]+mu_pad[1:-1,:-2]+mu_pad[1:-1,2:]-4.0*mu_a_2d)
        update = np.zeros_like(mu_a_2d)
        update[~obs_mask_2d] = kappa * laplacian[~obs_mask_2d]
        mu_a_2d = mu_a_2d + 0.1 * update
    return mu_a_2d

t0 = time.perf_counter()
for t in eval_t:
    fc = chl_norm[t]; fp = chl_norm[t-1]
    valid_mask = ~np.isnan(fc)
    obs_mask = valid_mask & (~held_out_20[t])
    y_obs = np.where(obs_mask, fc, 0.0)
    mu_b = fp.copy(); mu_b[np.isnan(mu_b)] = 0.0
    cellnn_da_2d(mu_b, y_obs, obs_mask)
results['Cell-NN pipeline'] = time.perf_counter() - t0
print(f"  -> {results['Cell-NN pipeline']:.2f}s")

# ── OI (L=0.2, fixed) ────────────────────────────────────────────────
print("\nTiming OI (L=0.2)...")
LAT_grid, LON_grid = LAT, LON
t0 = time.perf_counter()
for t in eval_t:
    fc = chl_norm[t]
    valid_mask = ~np.isnan(fc)
    obs_mask = valid_mask & (~held_out_20[t])
    tm = held_out_20[t]
    obs_rows, obs_cols = np.where(obs_mask)
    tst_rows, tst_cols = np.where(tm)
    if len(obs_rows) < 10 or len(tst_rows) == 0:
        continue
    obs_coords = np.stack([obs_rows/LAT_grid, obs_cols/LON_grid], axis=1)
    tst_coords = np.stack([tst_rows/LAT_grid, tst_cols/LON_grid], axis=1)
    obs_vals = fc[obs_rows, obs_cols]
    n_obs_use = min(800, len(obs_rows))
    if len(obs_rows) > n_obs_use:
        sel = np.random.choice(len(obs_rows), n_obs_use, replace=False)
        obs_coords = obs_coords[sel]; obs_vals = obs_vals[sel]
    D_oo = cdist(obs_coords, obs_coords); D_to = cdist(tst_coords, obs_coords)
    C_oo = np.exp(-0.5*(D_oo/0.2)**2) + 1e-4*np.eye(len(obs_coords))
    C_to = np.exp(-0.5*(D_to/0.2)**2)
    try:
        C_to @ np.linalg.solve(C_oo, obs_vals)
    except Exception:
        pass
results['OI'] = time.perf_counter() - t0
print(f"  -> {results['OI']:.2f}s")

# ── DINEOF (r=17, final reconstruction only, no CV) ────────────────
print("\nTiming DINEOF (r=17, final reconstruction only)...")
ever_valid = np.any(~np.isnan(chl_norm), axis=0)
idx_r, idx_c = np.where(ever_valid)
X_mat = chl_norm[:, idx_r, idx_c].copy()
held_out_mat = held_out_20[:, idx_r, idx_c]
X_work = X_mat.copy(); X_work[held_out_mat] = np.nan

def dineof_fill(X_in, r, tol=1e-4, max_iter=200):
    nan_mask = np.isnan(X_in)
    X = X_in.copy()
    col_mean = np.nanmean(X, axis=0); col_mean[np.isnan(col_mean)] = 0.0
    X[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    for it in range(max_iter):
        r_eff = min(r, min(X.shape) - 1)
        try:
            U, s, Vt = svds(X, k=r_eff)
        except Exception:
            break
        order = np.argsort(s)[::-1]; U, s, Vt = U[:, order], s[order], Vt[order, :]
        X_low = (U * s) @ Vt
        X_new = X.copy(); X_new[nan_mask] = X_low[nan_mask]
        change = np.linalg.norm(X_new - X) / max(np.linalg.norm(X), 1e-10)
        X = X_new
        if change < tol:
            break
    return X

t0 = time.perf_counter()
dineof_fill(X_work, 17, tol=1e-4, max_iter=200)
results['DINEOF'] = time.perf_counter() - t0
print(f"  -> {results['DINEOF']:.2f}s")

# ── Monte Carlo (causal, per-pixel fit + 10,000 draws) ──────────────
print("\nTiming Monte Carlo (per-pixel fit, full grid)...")
obs_mask_all = (~np.isnan(chl_norm)) & (~held_out_20)
train_stack = np.where(obs_mask_all, chl_raw, np.nan)
N_DRAWS = 10000
t0 = time.perf_counter()
n_fitted = 0
for i in range(LAT):
    for j in range(LON):
        series = train_stack[:, i, j]
        vals = series[~np.isnan(series)]
        if len(vals) < 10:
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
            dist.rvs(*best_params, size=N_DRAWS)
        else:
            try:
                kde = stats.gaussian_kde(vals)
                kde.resample(N_DRAWS)
            except Exception:
                pass
        n_fitted += 1
    if i % 100 == 0:
        print(f"  row {i}/{LAT}...")
results['Monte Carlo'] = time.perf_counter() - t0
print(f"  -> {results['Monte Carlo']:.2f}s ({n_fitted} pixels fitted)")

print(f"\n{'='*65}")
print("SATELLITE PIPELINE COST COMPARISON (single-pass, full 229 timesteps)")
print(f"{'='*65}")
print(f"{'Method':<20}{'Wall time (s)':>15}{'Per-timestep (s)':>20}")
print("-"*65)
for method, t in results.items():
    per_t = t / len(eval_t)
    print(f"{method:<20}{t:>15.2f}{per_t:>20.4f}")

np.savez('results/satellite_cost_comparison.npz',
         methods=list(results.keys()), times=np.array(list(results.values())))
print("\nSaved -> results/satellite_cost_comparison.npz")
print("Done.")
