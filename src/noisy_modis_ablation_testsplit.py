"""
Noisy MODIS Observation Ablation -- TEST SPLIT (leakage fix)
================================================================
Identical logic to src/noisy_modis_ablation.py, EXCEPT:
  1. kappa=0.4, n_diff=50 now passed EXPLICITLY at every call site
     (matching Table 6/8/9's paper-wide setting -- the original
     script's kappa=0.3, n_diff=30 default was stale and unused
     intentionally elsewhere, confirmed with user).
  2. obs_mask excludes the FULL held_out_20 (unchanged reconstruction
     input), scoring is against test_mask only (not the full 20%).
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.da_cellnn import cellnn_da_step
import warnings
warnings.filterwarnings('ignore')

chl_norm = np.load('data/modis_chl/chl_norm.npy')
N, LAT, LON = chl_norm.shape

split = np.load('results/holdout_split_masks.npz')
held_out_20 = split['held_out_20']
test_mask   = split['test_mask']
eval_t      = list(split['eval_t'])

NOISE_LEVELS = [0.01, 0.05, 0.10, 0.20, 0.30]
NOISE_SEEDS  = [42, 43, 44]
NOISE_LABELS = ['1%', '5%', '10%', '20%', '30%']

print(f"Test timesteps: {len(eval_t)} | Test pixels: {int(test_mask.sum()):,}")

def reconstruct_2d(mu_b_2d, y_obs_2d, obs_mask_2d,
                   use_cellnn, use_diffusion,
                   alpha_A=1.0, wb=1.0, kappa=0.4, n_diff=50):
    lat_n, lon_n = mu_b_2d.shape
    mu_b_flat  = mu_b_2d.flatten()
    y_obs_flat = y_obs_2d.flatten()
    obs_flat   = obs_mask_2d.flatten()
    mu_a_flat  = mu_b_flat.copy()

    if use_cellnn:
        mu_a_obs, _ = cellnn_da_step(
            mu_b_flat[obs_flat], y_obs_flat[obs_flat],
            alpha_A=alpha_A, wb=wb, d_tau=5e-2,
            tau_max=1e-1, eps_conv=1e-6, r_max=50)
        mu_a_flat[obs_flat] = mu_a_obs
    else:
        mu_a_flat[obs_flat] = y_obs_flat[obs_flat]

    mu_a_2d = mu_a_flat.reshape(lat_n, lon_n)

    if use_diffusion:
        for _ in range(n_diff):
            p = np.pad(mu_a_2d, 1, mode='edge')
            L = (p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*mu_a_2d)
            u = np.zeros_like(mu_a_2d)
            u[~obs_mask_2d] = kappa * L[~obs_mask_2d]
            mu_a_2d = mu_a_2d + 0.1 * u

    return mu_a_2d

CONFIGS = {
    'A_Background':        dict(use_cellnn=False, use_diffusion=False),
    'B_Diffusion_only':    dict(use_cellnn=False, use_diffusion=True),
    'C_CellNN_only':       dict(use_cellnn=True,  use_diffusion=False),
    'D_CellNN_Diffusion':  dict(use_cellnn=True,  use_diffusion=True),
}
FIXED = dict(kappa=0.4, n_diff=50)

print("\nRunning noise-free reference (sigma=0.00)...")
ref_rmse = {cfg: [] for cfg in CONFIGS}
for t in eval_t:
    fc = chl_norm[t]; fp = chl_norm[t-1]
    valid_mask = ~np.isnan(fc)
    obs_mask = valid_mask & (~held_out_20[t])
    tm = test_mask[t]
    mu_b = fp.copy(); mu_b[np.isnan(mu_b)] = 0.0
    y_obs = np.where(obs_mask, fc, 0.0)
    rows, cols = np.where(tm)
    if len(rows) == 0:
        continue
    true_v = fc[rows, cols]
    for cfg, kwargs in CONFIGS.items():
        mu_a = reconstruct_2d(mu_b, y_obs, obs_mask, **kwargs, **FIXED)
        ref_rmse[cfg].append(np.sqrt(np.mean((true_v - mu_a[rows,cols])**2)))

for cfg in CONFIGS:
    ref_rmse[cfg] = np.mean(ref_rmse[cfg])
print(f"  Noise-free RMSE: "
      f"BG={ref_rmse['A_Background']:.4f} | "
      f"Diff={ref_rmse['B_Diffusion_only']:.4f} | "
      f"CNN={ref_rmse['C_CellNN_only']:.4f} | "
      f"CNN+Diff={ref_rmse['D_CellNN_Diffusion']:.4f}")

results = {}
for sigma, label in zip(NOISE_LEVELS, NOISE_LABELS):
    print(f"\nNoise sigma={sigma} ({label})...")
    seed_rmses = {cfg: [] for cfg in CONFIGS}
    for nseed in NOISE_SEEDS:
        rng = np.random.default_rng(nseed)
        per_t = {cfg: [] for cfg in CONFIGS}
        for t in eval_t:
            fc = chl_norm[t]; fp = chl_norm[t-1]
            valid_mask = ~np.isnan(fc)
            obs_mask = valid_mask & (~held_out_20[t])
            tm = test_mask[t]
            mu_b = fp.copy(); mu_b[np.isnan(mu_b)] = 0.0
            noise_field = np.zeros_like(fc)
            n_obs = obs_mask.sum()
            noise_field[obs_mask] = rng.normal(0, sigma, size=n_obs)
            y_obs_noisy = np.where(obs_mask, fc + noise_field, 0.0)
            rows, cols = np.where(tm)
            if len(rows) == 0:
                continue
            true_v = fc[rows, cols]
            for cfg, kwargs in CONFIGS.items():
                mu_a = reconstruct_2d(mu_b, y_obs_noisy, obs_mask, **kwargs, **FIXED)
                per_t[cfg].append(np.sqrt(np.mean((true_v - mu_a[rows,cols])**2)))
        for cfg in CONFIGS:
            seed_rmses[cfg].append(np.mean(per_t[cfg]))

    results[sigma] = {}
    for cfg in CONFIGS:
        m = np.mean(seed_rmses[cfg])
        s = np.std(seed_rmses[cfg])
        results[sigma][cfg] = (m, s)
        print(f"  {cfg:<25} RMSE={m:.4f} ± {s:.4f}")

print(f"\n{'='*80}")
print("NOISY MODIS ABLATION (TEST SPLIT) -- SUMMARY TABLE")
print(f"{'='*80}")
print(f"{'sigma':>6}  {'label':>5}  {'Background':>12}  {'Diffusion':>12}  {'Cell-NN':>12}  {'CNN+Diff':>12}")
print("-"*80)
print(f"{'0.00':>6}  {'0%':>5}  "
      f"{ref_rmse['A_Background']:>12.4f}  "
      f"{ref_rmse['B_Diffusion_only']:>12.4f}  "
      f"{ref_rmse['C_CellNN_only']:>12.4f}  "
      f"{ref_rmse['D_CellNN_Diffusion']:>12.4f}  (noise-free reference)")
for sigma, label in zip(NOISE_LEVELS, NOISE_LABELS):
    bg = results[sigma]['A_Background']; df = results[sigma]['B_Diffusion_only']
    cn = results[sigma]['C_CellNN_only']; cd = results[sigma]['D_CellNN_Diffusion']
    diff_vs_diff = cd[0] - df[0]
    print(f"{sigma:>6.2f}  {label:>5}  "
          f"{bg[0]:>7.4f}±{bg[1]:.4f}  {df[0]:>7.4f}±{df[1]:.4f}  "
          f"{cn[0]:>7.4f}±{cn[1]:.4f}  {cd[0]:>7.4f}±{cd[1]:.4f}  "
          f"(CNN+Diff vs Diff: {diff_vs_diff:+.4f})")
print(f"{'='*80}")
print("[Compare to original Table 11: ref=0.1335, 30%=0.1971±0.0001]")

np.savez('results/noisy_modis_ablation_testsplit.npz',
         noise_levels=np.array(NOISE_LEVELS), noise_labels=NOISE_LABELS,
         ref_rmse_bg=ref_rmse['A_Background'], ref_rmse_diff=ref_rmse['B_Diffusion_only'],
         ref_rmse_cnn=ref_rmse['C_CellNN_only'], ref_rmse_cnndiff=ref_rmse['D_CellNN_Diffusion'],
         results_bg=np.array([(results[s]['A_Background']) for s in NOISE_LEVELS]),
         results_diff=np.array([(results[s]['B_Diffusion_only']) for s in NOISE_LEVELS]),
         results_cnn=np.array([(results[s]['C_CellNN_only']) for s in NOISE_LEVELS]),
         results_cnndiff=np.array([(results[s]['D_CellNN_Diffusion']) for s in NOISE_LEVELS]))
print("\nSaved -> results/noisy_modis_ablation_testsplit.npz")
print("Done.")
