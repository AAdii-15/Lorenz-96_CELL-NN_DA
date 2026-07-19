"""
Cell-NN Under Realistic Observation Noise
=========================================================================
Tests the actual DA task Cell-NN is designed for: blending a NOISY
observation with a background forecast, rather than the noise-free
direct-copy scenario in Experiment 1 (where Cell-NN's relaxation was
shown to add zero value, since its fixed point is exactly the clean
observation with nothing to "blend").

Noise level (std=0.2394) is not arbitrary -- it matches the empirically
measured MODIS-VIIRS RMS difference from the external validation,
i.e. a realistic proxy for genuine cross-sensor/retrieval uncertainty,
not a value chosen to favor any particular outcome.

Run across 5 independent noise realizations (seeds 42-46) to report
mean +/- std, partially addressing the reviewer's separate concern
about missing statistical significance / uncertainty quantification.
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath('.')))
from src.da_cellnn import cellnn_da_step
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Cell-NN Under Realistic Observation Noise")
print("=" * 70)

chl_norm = np.load('data/modis_chl/chl_norm.npy')
N, LAT, LON = chl_norm.shape

NOISE_STD = 0.2394
SEEDS = [42, 43, 44, 45, 46]

print(f"Injected noise std: {NOISE_STD} (matches empirical MODIS-VIIRS RMS difference)")
print(f"Seeds: {SEEDS}")

all_results = {}

for seed in SEEDS:
    np.random.seed(seed)
    rmse_raw_noisy, rmse_cellnn, rmse_background, n_iters_all = [], [], [], []

    for t in range(1, N):
        field_curr = chl_norm[t]
        field_prev = chl_norm[t - 1]
        valid_mask = ~np.isnan(field_curr)
        if valid_mask.sum() < 500:
            continue

        mu_b = field_prev.copy()
        mu_b[np.isnan(mu_b)] = 0.0

        noise = np.random.normal(0, NOISE_STD, size=field_curr.shape)
        y_obs_noisy_field = field_curr + noise
        y_obs_noisy_field[~valid_mask] = 0.0

        mu_b_flat = mu_b.flatten()
        y_flat = y_obs_noisy_field.flatten()
        obs_flat = valid_mask.flatten()
        mu_a_obs, n_iter = cellnn_da_step(mu_b_flat[obs_flat], y_flat[obs_flat],
                                           alpha_A=1.0, wb=1.0, d_tau=5e-2,
                                           tau_max=1e-1, eps_conv=1e-6, r_max=50)

        true_v = field_curr[valid_mask]
        noisy_v = y_obs_noisy_field[valid_mask]
        bg_v = mu_b[valid_mask]

        rmse_raw_noisy.append(np.sqrt(np.mean((true_v - noisy_v) ** 2)))
        rmse_cellnn.append(np.sqrt(np.mean((true_v - mu_a_obs) ** 2)))
        rmse_background.append(np.sqrt(np.mean((true_v - bg_v) ** 2)))
        n_iters_all.append(n_iter)

    all_results[seed] = {
        'raw_noisy': np.mean(rmse_raw_noisy),
        'cellnn': np.mean(rmse_cellnn),
        'background': np.mean(rmse_background),
        'mean_iters': np.mean(n_iters_all),
    }
    print(f"  seed={seed} | RawNoisy={all_results[seed]['raw_noisy']:.4f} | "
          f"Cell-NN={all_results[seed]['cellnn']:.4f} | "
          f"Background={all_results[seed]['background']:.4f} | "
          f"mean_iters={all_results[seed]['mean_iters']:.1f}")

raw_vals = np.array([all_results[s]['raw_noisy'] for s in SEEDS])
cellnn_vals = np.array([all_results[s]['cellnn'] for s in SEEDS])
bg_vals = np.array([all_results[s]['background'] for s in SEEDS])
iters_vals = np.array([all_results[s]['mean_iters'] for s in SEEDS])

print(f"\n{'=' * 70}")
print("SUMMARY -- Mean +/- Std across 5 noise realizations")
print(f"{'=' * 70}")
print(f"{'Method':<25}{'RMSE (mean)':>15}{'RMSE (std)':>15}")
print("-" * 70)
print(f"{'Raw noisy observation':<25}{raw_vals.mean():>15.4f}{raw_vals.std():>15.4f}")
print(f"{'Cell-NN (blended)':<25}{cellnn_vals.mean():>15.4f}{cellnn_vals.std():>15.4f}")
print(f"{'Raw background':<25}{bg_vals.mean():>15.4f}{bg_vals.std():>15.4f}")
print(f"{'-' * 70}")
print(f"Mean DA iterations: {iters_vals.mean():.1f} (out of r_max=50)")
print(f"{'=' * 70}")

if cellnn_vals.mean() < raw_vals.mean() and cellnn_vals.mean() < bg_vals.mean():
    print("\nRESULT: Cell-NN beats BOTH raw noisy observation and raw background.")
    print("This IS genuine evidence of DA blending value under realistic noise.")
elif cellnn_vals.mean() < raw_vals.mean():
    print("\nRESULT: Cell-NN beats raw noisy observation but not raw background.")
elif cellnn_vals.mean() < bg_vals.mean():
    print("\nRESULT: Cell-NN beats raw background but not raw noisy observation.")
else:
    print("\nRESULT: Cell-NN does not beat either baseline.")
    print("This would indicate the relaxation step adds no value even under noise.")

np.savez('results/noisy_observation_test.npz',
         raw_vals=raw_vals, cellnn_vals=cellnn_vals, bg_vals=bg_vals,
         iters_vals=iters_vals, noise_std=NOISE_STD, seeds=SEEDS)
print("\nSaved -> results/noisy_observation_test.npz")
print("\nDone.")
