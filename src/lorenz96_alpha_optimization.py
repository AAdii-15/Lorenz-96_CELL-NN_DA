"""
Lorenz-96 alpha_A optimization: detailed analysis.
=====================================================
Reuses the EXACT primary realization parameters (K, F, dt, N,
da_interval, da_on_periods) from lorenz96_da.py. R-score is computed
inline (NOT imported from intermittency_gap_length.py, which has no
__main__ guard and re-runs its entire 30-simulation experiment as a
side effect on import -- the compute_r_score logic is copied here
verbatim instead, avoiding that entirely).

For each alpha_A in a swept range, records:
  1. R-score, AVERAGED OVER 5 SEEDS (same seed pairs used in the
     paper's own gap-length robustness check), matching the
     multi-realization approach the paper itself uses for
     robustness claims -- not a single noisy realization.
  2. Average relaxation iterations used per DA cycle.
  3. Single-step relaxation error (matching Figure 2's metric, for
     a dedicated Lorenz-96-only comparison).
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from src.lorenz96 import (integrate_lorenz96, make_initial_condition,
                           generate_observations_96)
from src.lorenz96_da import run_cellnn_lorenz96
from src.da_cellnn import cellnn_da_step

print("="*65)
print("Lorenz-96 alpha_A optimization: detailed sweep")
print("="*65)

K, F, dt, N = 40, 8.0, 1e-2, 5000
da_interval = 25
da_on_periods = [(0, 2000), (3500, 5000)]

# Inline copy of intermittency_gap_length.py's compute_r_score --
# not imported, to avoid triggering that file's unguarded top-level
# experiment code.
def compute_r_score(X_true, X_pred, da_on_periods, N):
    mask = np.zeros(N, dtype=bool)
    for s, e in da_on_periods:
        mask[s:e] = True
    if mask.sum() == 0:
        return np.nan
    return np.mean((X_true[mask] - X_pred[mask])**2)

# Same 5 seed pairs used in the paper's own gap-length robustness
# check (intermittency_gap_length.py), for consistency with how
# multi-realization robustness is established elsewhere in the paper.
SEEDS = [(42,123), (13,31), (99,17), (55,88), (101,202)]

alpha_values = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]

results = []
print(f"\nSweeping alpha_A across {len(alpha_values)} values, "
      f"{len(SEEDS)} seeds each ({len(alpha_values)*len(SEEDS)} total runs)...")
print(f"{'alpha_A':>8} {'R-score(mean)':>14} {'R-score(std)':>13} {'Avg iters':>10}")
print("-"*50)

for alpha_A in alpha_values:
    r_scores = []
    iter_counts = []
    for true_seed, bg_seed in SEEDS:
        X0_true = make_initial_condition(K, F, perturb=0.0, seed=true_seed)
        X0_bg   = make_initial_condition(K, F, perturb=0.05, seed=bg_seed)
        _, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt)
        np.random.seed(true_seed)
        X_obs, obs_idx = generate_observations_96(X_true, noise_level=0.05, obs_every=2)

        X_cnn, errors_cnn = run_cellnn_lorenz96(
            X_true, X_obs, obs_idx, X0_bg, dt,
            da_interval=da_interval, da_on_periods=da_on_periods,
            alpha_A=alpha_A, wb=1.0, K=K, F=F
        )
        r_scores.append(compute_r_score(X_true, X_cnn, da_on_periods, N))
        iter_counts.append(np.mean([e['n_iter'] for e in errors_cnn]))

    r_mean = np.mean(r_scores)
    r_std  = np.std(r_scores)
    iter_mean = np.mean(iter_counts)

    results.append({
        'alpha_A': alpha_A,
        'r_score_mean': r_mean,
        'r_score_std': r_std,
        'r_scores_all_seeds': r_scores,
        'avg_iter': iter_mean
    })
    print(f"{alpha_A:8.2f} {r_mean:14.4f} {r_std:13.4f} {iter_mean:10.1f}")

# ── Single-relaxation-step error, for the dedicated L96-only comparison ──
print("\nComputing single-relaxation-step error (matching Figure 2's metric)...")
np.random.seed(0)
mu_b_test = np.random.randn(1000) * 5
mu_obs_test = mu_b_test + np.random.randn(1000) * 0.5

single_step_results = []
for alpha_A in alpha_values:
    mu_a, n_iter = cellnn_da_step(mu_b_test, mu_obs_test, alpha_A=alpha_A, wb=1.0)
    err = np.mean(np.abs(mu_a - mu_obs_test))
    single_step_results.append({'alpha_A': alpha_A, 'error': err, 'n_iter': n_iter})
    print(f"  alpha_A={alpha_A:.2f}: single-step error={err:.6f}, iters={n_iter}")

np.save('results/lorenz96_alpha_sweep.npy', results)
np.save('results/lorenz96_alpha_single_step.npy', single_step_results)

print("\nSweep complete. Results saved -> results/lorenz96_alpha_sweep.npy")
