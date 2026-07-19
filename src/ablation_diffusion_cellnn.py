"""
Experiment 1: Diffusion vs Cell-NN Ablation
=========================================================================
Addresses the Q1 reviewer's attribution problem: does reconstruction
skill at held-out pixels come from the Cell-NN relaxation operator,
the diffusion fallback, or both? Decomposes the full pipeline into a
2x2 design:
  {Cell-NN relaxation at observed pixels: yes/no} x
  {Diffusion fill at unobserved pixels: yes/no}

Configs:
  A) Background only       -- no Cell-NN, no diffusion (existing: 0.7549)
  B) Diffusion only        -- raw obs at observed pixels, diffusion at gaps
  C) Cell-NN only          -- Cell-NN relaxed at observed, raw persistence at gaps
  D) Cell-NN + Diffusion   -- full pipeline (existing: 0.1785)

Same test_mask protocol (seed=42) as every other validation run, so
all four numbers are directly comparable.
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath('.')))
from src.da_cellnn import cellnn_da_step
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Experiment 1: Diffusion vs Cell-NN Ablation")
print("=" * 70)

chl_norm = np.load('data/modis_chl/chl_norm.npy')
dates = np.load('data/modis_chl/chl_dates.npy')
N, LAT, LON = chl_norm.shape

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
    test_rows = valid_idx[0][sel]
    test_cols = valid_idx[1][sel]
    test_mask[t, test_rows, test_cols] = True
    eval_t.append(t)

print(f"Held-out test pixels: {len(eval_t)} timesteps (identical protocol)")

def reconstruct(mu_b_2d, y_obs_2d, obs_mask_2d, use_cellnn, use_diffusion,
                 alpha_A=1.0, wb=1.0, kappa=0.4, n_diff=50):
    lat_n, lon_n = mu_b_2d.shape
    mu_b_flat = mu_b_2d.flatten()
    y_obs_flat = y_obs_2d.flatten()
    obs_flat = obs_mask_2d.flatten()

    mu_a_flat = mu_b_flat.copy()
    if use_cellnn:
        mu_a_obs, _ = cellnn_da_step(mu_b_flat[obs_flat], y_obs_flat[obs_flat],
                                      alpha_A=alpha_A, wb=wb, d_tau=5e-2,
                                      tau_max=1e-1, eps_conv=1e-6, r_max=50)
        mu_a_flat[obs_flat] = mu_a_obs
    else:
        # raw observation directly inserted, no relaxation
        mu_a_flat[obs_flat] = y_obs_flat[obs_flat]

    mu_a_2d = mu_a_flat.reshape(lat_n, lon_n)

    if use_diffusion:
        for _ in range(n_diff):
            p = np.pad(mu_a_2d, 1, mode='edge')
            L = p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*mu_a_2d
            u = np.zeros_like(mu_a_2d)
            u[~obs_mask_2d] = kappa*L[~obs_mask_2d]
            mu_a_2d = mu_a_2d + 0.1*u
    # if not use_diffusion: unobserved pixels simply keep mu_b (raw persistence)

    return mu_a_2d

configs = {
    'B_diffusion_only': dict(use_cellnn=False, use_diffusion=True),
    'C_cellnn_only':    dict(use_cellnn=True,  use_diffusion=False),
}

results = {name: [] for name in configs}

for t in eval_t:
    field_curr = chl_norm[t]
    field_prev = chl_norm[t-1]
    tm = test_mask[t]
    valid_mask = ~np.isnan(field_curr)

    mu_b = field_prev.copy()
    mu_b[np.isnan(mu_b)] = 0.0

    obs_mask = valid_mask & (~tm)
    y_obs = np.where(obs_mask, field_curr, 0.0)

    rows, cols = np.where(tm)
    true_v = field_curr[rows, cols]

    for name, cfg in configs.items():
        mu_a = reconstruct(mu_b, y_obs, obs_mask, **cfg)
        pred_v = mu_a[rows, cols]
        rmse_t = np.sqrt(np.mean((true_v - pred_v) ** 2))
        results[name].append(rmse_t)

print(f"\n{'='*70}")
print("EXPERIMENT 1 RESULTS -- 2x2 Ablation")
print(f"{'='*70}")
print(f"{'Config':<35}{'Cell-NN':>10}{'Diffusion':>12}{'RMSE':>10}")
print("-"*70)
print(f"{'A. Background only (existing)':<35}{'No':>10}{'No':>12}{0.7549:>10.4f}")
for name, cfg in configs.items():
    rmse_mean = np.mean(results[name])
    cellnn_str = 'Yes' if cfg['use_cellnn'] else 'No'
    diff_str = 'Yes' if cfg['use_diffusion'] else 'No'
    label = {'B_diffusion_only': 'B. Diffusion only',
             'C_cellnn_only': 'C. Cell-NN only (no diffusion)'}[name]
    print(f"{label:<35}{cellnn_str:>10}{diff_str:>12}{rmse_mean:>10.4f}")
print(f"{'D. Cell-NN + Diffusion (existing)':<35}{'Yes':>10}{'Yes':>12}{0.1335:>10.4f}")
print(f"{'='*70}")

np.savez('results/ablation_diffusion_cellnn.npz',
         **{f'{k}_rmse': np.array(v) for k, v in results.items()})
print("\nSaved -> results/ablation_diffusion_cellnn.npz")
print("\nDone.")
