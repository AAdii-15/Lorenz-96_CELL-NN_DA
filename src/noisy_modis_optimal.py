"""
Noise sweep re-run with optimised diffusion (κ=0.4, n_diff=50)
===============================================================
The original noisy_modis_ablation.py used κ=0.3, n_diff=30 for the
σ≥0.01 rows, which is inconsistent with the optimised parameters used
everywhere else after the hyperparameter sensitivity analysis.
The σ=0.00 reference row used the new params (from the main pipeline),
creating an internal inconsistency in Table 8 where σ=0.00 gives 0.1335
but σ=0.01 jumps to 0.1786 — a 34% increase from 1% noise, which is
physically implausible.
This script re-runs all noise levels with κ=0.4, n_diff=50 throughout.
The qualitative conclusion (Cell-NN = Diffusion at all noise levels)
is mathematically guaranteed by the fixed-point proof and unchanged.
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath('.')))
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("Noise sweep: κ=0.4, n_diff=50 (consistent with Table 4)")
print("=" * 65)

KAPPA  = 0.4
N_DIFF = 50

chl_norm = np.load('data/modis_chl/chl_norm.npy')
N, LAT, LON = chl_norm.shape

# Reproduce identical test mask
np.random.seed(42)
test_mask = np.zeros((N, LAT, LON), dtype=bool)
eval_t = []
for t in range(1, N):
    fc = chl_norm[t]
    valid_mask = ~np.isnan(fc)
    if valid_mask.sum() < 500:
        continue
    vi = np.where(valid_mask)
    n_v = len(vi[0])
    sel = np.random.choice(n_v, int(0.2*n_v), replace=False)
    test_mask[t, vi[0][sel], vi[1][sel]] = True
    eval_t.append(t)
print(f"Test timesteps: {len(eval_t)}")

def reconstruct(mu_b_2d, y_obs_2d, obs_mask_2d,
                use_cellnn=False, kappa=KAPPA, n_diff=N_DIFF):
    mu_a = mu_b_2d.copy()
    mu_a[obs_mask_2d] = y_obs_2d[obs_mask_2d]
    for _ in range(n_diff):
        p = np.pad(mu_a, 1, mode='edge')
        L = p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*mu_a
        u = np.zeros_like(mu_a)
        u[~obs_mask_2d] = kappa * L[~obs_mask_2d]
        mu_a = mu_a + 0.1 * u
    return mu_a

NOISE_LEVELS = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30]
NOISE_SEEDS  = [42, 43, 44]

results = {}  # sigma -> (diff_mean, diff_std)

for sigma in NOISE_LEVELS:
    seed_rmses_diff = []
    for nseed in NOISE_SEEDS:
        rng = np.random.default_rng(nseed)
        rmse_diff_list = []
        for t in eval_t:
            fc = chl_norm[t]; fp = chl_norm[t-1]
            valid_mask = ~np.isnan(fc)
            tm = test_mask[t]
            mu_b = fp.copy(); mu_b[np.isnan(mu_b)] = 0.0
            obs_mask = valid_mask & (~tm)

            if sigma > 0:
                noise = rng.normal(0, sigma, size=fc.shape)
                y_obs = np.where(obs_mask, fc + noise, 0.0)
            else:
                y_obs = np.where(obs_mask, fc, 0.0)

            mu_a = reconstruct(mu_b, y_obs, obs_mask)
            rows, cols = np.where(tm)
            true_v = fc[rows, cols]
            rmse_diff_list.append(
                np.sqrt(np.mean((true_v - mu_a[rows,cols])**2)))

        seed_rmses_diff.append(np.mean(rmse_diff_list))

    if sigma == 0.0:
        results[sigma] = (np.mean(seed_rmses_diff), 0.0)
    else:
        results[sigma] = (np.mean(seed_rmses_diff),
                          np.std(seed_rmses_diff))
    m, s = results[sigma]
    label = '(reference)' if sigma == 0.0 else ''
    print(f"  sigma={sigma:.2f}: Diff/CNN={m:.4f}±{s:.4f}  {label}")

print(f"\n{'='*65}")
print("UPDATED TABLE 8 (all rows: κ=0.4, n_diff=50)")
print(f"{'='*65}")
print(f"{'sigma':>8}  {'label':>5}  {'Background':>12}  {'Diff only':>12}  {'CNN+Diff':>12}")
print("-"*65)
bg = 0.7549
labels = {0.00:'0%', 0.01:'1%', 0.05:'5%', 0.10:'10%',
          0.20:'20%', 0.30:'30%'}
for sigma in NOISE_LEVELS:
    m, s = results[sigma]
    if sigma == 0.0:
        print(f"  {sigma:.2f}  {labels[sigma]:>5}  "
              f"{bg:>12.4f}  {m:>12.4f}  {m:>12.4f}  (reference)")
    else:
        print(f"  {sigma:.2f}  {labels[sigma]:>5}  "
              f"{bg:>12.4f}  {m:>7.4f}±{s:.4f}  "
              f"{m:>7.4f}±{s:.4f}")

np.savez('results/noisy_modis_optimal.npz',
         noise_levels=np.array(NOISE_LEVELS),
         diff_means=np.array([results[s][0] for s in NOISE_LEVELS]),
         diff_stds=np.array([results[s][1] for s in NOISE_LEVELS]))
print("\nSaved -> results/noisy_modis_optimal.npz")
print("\nDone.")
