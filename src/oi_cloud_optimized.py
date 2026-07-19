"""
OI Re-optimised for Cloud-Shaped Masking (Critical Fix #2)
============================================================
The original Table 12 used L=0.15 calibrated on RANDOM masking.
This is unfair under cloud-shaped masking where gaps span 100s of km.

This script:
  1. CV-optimises L on a held-out 30-pair validation set
     using cloud-shaped geometry (NOT the test pairs)
  2. Reports OI performance on the remaining 108 test pairs
     with the cloud-appropriate L
  3. Also reports the full L-vs-RMSE curve to show why
     L=0.15 fails and what the best achievable OI performance is

Key: we never use the 108 test pairs for L selection.
"""
import numpy as np
from scipy.spatial.distance import cdist
from datetime import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.abspath('.')))
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("OI Re-optimisation for Cloud-Shaped Masking")
print("CV on 30 validation pairs → Test on 108 held-out pairs")
print("=" * 70)

KAPPA, N_DIFF = 0.4, 50

chl_norm = np.load('data/modis_chl/chl_norm.npy')
dates    = np.load('data/modis_chl/chl_dates.npy')
N, LAT, LON = chl_norm.shape

def parse_date(ds):
    return datetime.strptime(str(ds), '%Y%m%d')
parsed = [parse_date(d) for d in dates]
years  = np.array([d.year for d in parsed])
doys   = np.array([d.timetuple().tm_yday for d in parsed])

# ── Rebuild same 138 eval pairs ─────────────────────────────────────────
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
print(f"Total evaluation pairs: {len(eval_pairs)}")

# Split: first 30 for CV, remaining 108 for test
np.random.seed(42)
idx = np.random.permutation(len(eval_pairs))
cv_idx   = idx[:30]
test_idx = idx[30:]
cv_pairs   = [eval_pairs[i] for i in cv_idx]
test_pairs = [eval_pairs[i] for i in test_idx]
print(f"CV pairs: {len(cv_pairs)} | Test pairs: {len(test_pairs)}")

# ── OI function ─────────────────────────────────────────────────────────
def oi_fill(obs_rows, obs_cols, obs_vals, tst_rows, tst_cols, L, 
            max_obs=500):
    """OI with Gaussian kernel, correlation length L (in normalised coords)."""
    if len(obs_rows) > max_obs:
        sel = np.random.choice(len(obs_rows), max_obs, replace=False)
        obs_rows = obs_rows[sel]; obs_cols = obs_cols[sel]
        obs_vals = obs_vals[sel]
    obs_c = np.stack([obs_rows/LAT, obs_cols/LON], axis=1)
    tst_c = np.stack([tst_rows/LAT, tst_cols/LON], axis=1)
    C_oo  = np.exp(-0.5*(cdist(obs_c, obs_c)/L)**2) + 1e-4*np.eye(len(obs_rows))
    C_to  = np.exp(-0.5*(cdist(tst_c, obs_c)/L)**2)
    try:
        pred = C_to @ np.linalg.solve(C_oo, obs_vals)
        return pred if np.isfinite(pred).all() else None
    except:
        return None

def eval_oi_pairs(pairs, L):
    """Evaluate OI with a given L on a set of pairs. Returns mean RMSE."""
    rmse_list = []
    np.random.seed(42)
    for ti, di in pairs:
        target = chl_norm[ti]; donor = chl_norm[di]
        gap_mask = np.isnan(donor) & (~np.isnan(target))
        if gap_mask.sum() < 50: continue
        obs_mask = (~np.isnan(target)) & (~np.isnan(donor))
        if obs_mask.sum() < 100: continue
        
        obs_rows, obs_cols = np.where(obs_mask)
        obs_vals = target[obs_rows, obs_cols]
        tst_rows, tst_cols = np.where(gap_mask)
        true_v = target[gap_mask]
        
        pred = oi_fill(obs_rows, obs_cols, obs_vals,
                       tst_rows, tst_cols, L)
        if pred is None: continue
        rmse_list.append(np.sqrt(np.mean((true_v - pred)**2)))
    return np.mean(rmse_list) if rmse_list else np.inf

# ── CV over L grid ───────────────────────────────────────────────────────
# Random masking optimal: L=0.15
# Cloud masking: gaps span 100-1000s of km → expect much larger L
# Normalised coordinates: LAT span ~17°, LON span ~20°
# L=0.15 in normalised = ~3° gap radius
# For 100km gaps (~1°), need L~0.1-0.3; for 1000km gaps (~10°), need L~1.0+
L_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]

print(f"\n--- CV phase: optimising L on {len(cv_pairs)} validation pairs ---")
print(f"{'L':>8}  {'CV RMSE':>12}")
print("-"*25)

cv_scores = {}
for L in L_GRID:
    rmse = eval_oi_pairs(cv_pairs, L)
    cv_scores[L] = rmse
    marker = ' <-- original (random-mask)' if L == 0.15 else ''
    print(f"  {L:>6.2f}  {rmse:>12.4f}{marker}")

best_L = min(cv_scores, key=cv_scores.get)
print(f"\nBest L (cloud-shaped CV): {best_L}")
print(f"CV RMSE at best L:        {cv_scores[best_L]:.4f}")
print(f"CV RMSE at original L=0.15: {cv_scores[0.15]:.4f}")

# ── Test phase ────────────────────────────────────────────────────────────
print(f"\n--- Test phase: {len(test_pairs)} held-out pairs ---")
print(f"Evaluating at best_L={best_L} and original L=0.15\n")

rmse_oi_best = eval_oi_pairs(test_pairs, best_L)
rmse_oi_orig = eval_oi_pairs(test_pairs, 0.15)

# Background RMSE on test pairs
bg_list = []
np.random.seed(42)
for ti, di in test_pairs:
    target = chl_norm[ti]; donor = chl_norm[di]
    gap_mask = np.isnan(donor) & (~np.isnan(target))
    if gap_mask.sum() < 50: continue
    mu_b = chl_norm[ti-1].copy(); mu_b[np.isnan(mu_b)] = 0.0
    bg_list.append(np.sqrt(np.mean((target[gap_mask]-mu_b[gap_mask])**2)))
rmse_bg = np.mean(bg_list)

# Also run diffusion pipeline on same test pairs
def diffusion_pipeline(mu_b, y_obs, obs_mask):
    mu_a = mu_b.copy()
    mu_a[obs_mask] = y_obs[obs_mask]
    for _ in range(N_DIFF):
        p = np.pad(mu_a, 1, mode='edge')
        L_lap = p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*mu_a
        u = np.zeros_like(mu_a)
        u[~obs_mask] = KAPPA * L_lap[~obs_mask]
        mu_a = mu_a + 0.1 * u
    return mu_a

diff_list = []
np.random.seed(42)
for ti, di in test_pairs:
    target = chl_norm[ti]; donor = chl_norm[di]
    gap_mask = np.isnan(donor) & (~np.isnan(target))
    if gap_mask.sum() < 50: continue
    obs_mask = (~np.isnan(target)) & (~np.isnan(donor))
    if obs_mask.sum() < 100: continue
    mu_b = chl_norm[ti-1].copy(); mu_b[np.isnan(mu_b)] = 0.0
    y_obs = np.where(obs_mask, target, 0.0)
    mu_a = diffusion_pipeline(mu_b, y_obs, obs_mask)
    diff_list.append(np.sqrt(np.mean((target[gap_mask]-mu_a[gap_mask])**2)))
rmse_diff = np.mean(diff_list)

print(f"{'='*70}")
print(f"CLOUD-SHAPED MASKING: FAIR OI COMPARISON")
print(f"{'='*70}")
print(f"{'Method':<35}{'RMSE':>10}{'Improvement':>14}")
print("-"*60)
print(f"{'Background (persistence)':<35}{rmse_bg:>10.4f}{'---':>14}")
print(f"{'OI (L=0.15, random-mask tuned)':<35}"
      f"{rmse_oi_orig:>10.4f}"
      f"{(rmse_bg-rmse_oi_orig)/rmse_bg*100:>13.1f}%")
print(f"{'OI (L={best_L}, cloud-mask CV)':<35}"
      f"{rmse_oi_best:>10.4f}"
      f"{(rmse_bg-rmse_oi_best)/rmse_bg*100:>13.1f}%")
print(f"{'Diffusion pipeline (κ=0.4, n=50)':<35}"
      f"{rmse_diff:>10.4f}"
      f"{(rmse_bg-rmse_diff)/rmse_bg*100:>13.1f}%")
print(f"{'='*70}")
print(f"\nKey finding:")
print(f"  OI with poorly-tuned L:  RMSE={rmse_oi_orig:.4f}")
print(f"  OI with CV-tuned L:      RMSE={rmse_oi_best:.4f}")
print(f"  Diffusion pipeline:      RMSE={rmse_diff:.4f}")
if rmse_oi_best < rmse_bg:
    print(f"  → Properly-tuned OI beats background "
          f"({(rmse_bg-rmse_oi_best)/rmse_bg*100:.1f}% improvement)")
    print(f"  → Diffusion pipeline still beats OI by "
          f"{(rmse_oi_best-rmse_diff)/rmse_oi_best*100:.1f}%")
else:
    print(f"  → Even CV-tuned OI cannot beat background under "
          f"cloud-shaped geometry")

np.savez('results/oi_cloud_optimized.npz',
         L_grid=np.array(L_GRID),
         cv_scores=np.array([cv_scores[L] for L in L_GRID]),
         best_L=best_L,
         rmse_bg=rmse_bg,
         rmse_oi_orig=rmse_oi_orig,
         rmse_oi_best=rmse_oi_best,
         rmse_diff=rmse_diff)
print("\nSaved -> results/oi_cloud_optimized.npz")
print("\nDone.")
