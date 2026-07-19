"""
Cloud-Gap Size vs RMSE Analysis
=================================
Uses already-saved cloud_mask_validation.npz per-pair results.
For each evaluation pair, bins by gap size (n test pixels) and 
shows how RMSE degrades as gaps become larger.
This directly answers: how does Laplacian diffusion degrade 
with increasing gap size?
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.abspath('.')))
from src.da_cellnn import cellnn_da_step
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("Experiment B: Cloud-Gap Size vs RMSE")
print("=" * 65)

chl_norm = np.load('data/modis_chl/chl_norm.npy')
dates    = np.load('data/modis_chl/chl_dates.npy')
N, LAT, LON = chl_norm.shape
date_strs = [str(d) for d in dates]

from datetime import datetime
def parse_date(ds):
    return datetime.strptime(str(ds), '%Y%m%d')

parsed = [parse_date(d) for d in dates]
years  = np.array([d.year for d in parsed])
doys   = np.array([d.timetuple().tm_yday for d in parsed])

# Rebuild eval pairs (same logic as cloud_mask_validation.py)
DOY_WINDOW = 8
eval_pairs = []
for ti in range(1, N):
    if years[ti] not in [2017, 2018, 2019]:
        continue
    target_doy = doys[ti]
    candidates = [di for di in range(N) 
                  if years[di] in [2015, 2016] 
                  and abs(doys[di] - target_doy) <= DOY_WINDOW]
    if not candidates:
        continue
    best = min(candidates, key=lambda di: abs(doys[di] - target_doy))
    eval_pairs.append((ti, best))

print(f"Evaluation pairs: {len(eval_pairs)}")

def reconstruct_diff(mu_b_2d, y_obs_2d, obs_mask_2d,
                     kappa=0.3, n_diff=30):
    mu_a = mu_b_2d.copy()
    mu_a[obs_mask_2d] = y_obs_2d[obs_mask_2d]
    for _ in range(n_diff):
        p = np.pad(mu_a, 1, mode='edge')
        L = p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*mu_a
        u = np.zeros_like(mu_a)
        u[~obs_mask_2d] = kappa * L[~obs_mask_2d]
        mu_a = mu_a + 0.1 * u
    return mu_a

gap_sizes, rmse_bg_list, rmse_cnn_list = [], [], []

print("Computing per-pair RMSE and gap size...")
for ti, di in eval_pairs:
    target = chl_norm[ti]
    donor  = chl_norm[di]
    cloud_gap_mask = np.isnan(donor) & (~np.isnan(target))
    n_test = int(cloud_gap_mask.sum())
    if n_test < 50:
        continue
    obs_mask = (~np.isnan(target)) & (~np.isnan(donor))
    if obs_mask.sum() < 200:
        continue
    mu_b = chl_norm[ti - 1].copy()
    mu_b[np.isnan(mu_b)] = 0.0
    y_obs = np.where(obs_mask, target, 0.0)
    mu_a = reconstruct_diff(mu_b, y_obs, obs_mask)
    true_v = target[cloud_gap_mask]
    bg_v   = mu_b[cloud_gap_mask]
    pred_v = mu_a[cloud_gap_mask]
    gap_sizes.append(n_test)
    rmse_bg_list.append(np.sqrt(np.mean((true_v - bg_v)**2)))
    rmse_cnn_list.append(np.sqrt(np.mean((true_v - pred_v)**2)))

gap_sizes    = np.array(gap_sizes)
rmse_bg_list = np.array(rmse_bg_list)
rmse_cnn_list= np.array(rmse_cnn_list)

print(f"Gap size range: {gap_sizes.min():,} – {gap_sizes.max():,} pixels")

# Bin by gap size into quartiles
quartiles = np.percentile(gap_sizes, [0, 25, 50, 75, 100])
bin_labels = ['Q1 (small)', 'Q2 (medium)', 'Q3 (large)', 'Q4 (very large)']
print(f"\nGap size quartile boundaries: {quartiles.astype(int)}")

print(f"\n{'Bin':<20}{'Gap size range':>20}{'n':>6}{'BG RMSE':>12}{'Diff RMSE':>12}{'Improv%':>10}")
print("-"*82)

bin_results = []
for i in range(4):
    lo, hi = quartiles[i], quartiles[i+1]
    sel = (gap_sizes >= lo) & (gap_sizes <= hi)
    if sel.sum() == 0:
        continue
    bg_m  = rmse_bg_list[sel].mean()
    cnn_m = rmse_cnn_list[sel].mean()
    imp   = (bg_m - cnn_m) / bg_m * 100
    size_range = f"{int(lo):,}–{int(hi):,}"
    print(f"{bin_labels[i]:<20}{size_range:>20}{sel.sum():>6}{bg_m:>12.4f}"
          f"{cnn_m:>12.4f}{imp:>9.1f}%")
    bin_results.append((bin_labels[i], int(lo), int(hi), int(sel.sum()), 
                        bg_m, cnn_m, imp))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(gap_sizes/1000, rmse_cnn_list, alpha=0.5, s=20,
                color='#2196F3', label='Diffusion RMSE')
axes[0].scatter(gap_sizes/1000, rmse_bg_list, alpha=0.3, s=20,
                color='#F44336', label='Background RMSE')
axes[0].set_xlabel('Cloud gap size (thousand pixels)', fontsize=11)
axes[0].set_ylabel('RMSE (normalized scale)', fontsize=11)
axes[0].set_title('RMSE vs Cloud Gap Size\n(per target/donor pair)',
                  fontsize=11, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

improv_vals = (rmse_bg_list - rmse_cnn_list) / rmse_bg_list * 100
axes[1].scatter(gap_sizes/1000, improv_vals, alpha=0.5, s=20,
                color='#4CAF50')
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No improvement')
axes[1].set_xlabel('Cloud gap size (thousand pixels)', fontsize=11)
axes[1].set_ylabel('Improvement over background (%)', fontsize=11)
axes[1].set_title('Improvement % vs Cloud Gap Size',
                  fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Laplacian Diffusion Performance vs Contiguous Cloud Gap Size\n'
             'Bay of Bengal MODIS Chl-a (2017–2019 targets, 2015–2016 donors)',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('results/cloud_gap_size_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nSaved -> results/cloud_gap_size_analysis.png")

np.savez('results/cloud_gap_size_analysis.npz',
         gap_sizes=gap_sizes, rmse_bg=rmse_bg_list, rmse_diff=rmse_cnn_list,
         quartiles=quartiles)
print("Saved -> results/cloud_gap_size_analysis.npz")
print("\nDone.")
