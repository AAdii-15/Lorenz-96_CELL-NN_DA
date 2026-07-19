"""
Lorenz-96 DA — FINAL VERSION
αA=1.0, wb=1.0 (optimized from grid search)
noise_level = noise (FIXED — was noise/2 before)
da_interval = 25
perturb     = 5%
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lorenz96    import (lorenz96_fast, integrate_lorenz96,
                              make_initial_condition,
                              generate_observations_96)
from src.da_cellnn   import cellnn_da_step
from src.lorenz96_da import (run_cellnn_lorenz96, run_3dvar_lorenz96)

# ── Setup ────────────────────────────────────────────
K   = 40
F   = 8.0
dt  = 1e-2
N   = 5000

X0_true = make_initial_condition(K, F, perturb=0.0,  seed=42)
X0_bg   = make_initial_condition(K, F, perturb=0.05, seed=123)  # 5% perturbation

da_interval   = 25
da_on_periods = [(0, 2000), (3500, 5000)]

# FINAL OPTIMIZED PARAMETERS (from grid search)
ALPHA_A = 1.0   # was 0.1 in paper, 1.0 is our optimized value
WB      = 1.0   # was 3.0 in paper, 1.0 is our optimized value

t, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt)

# ── Different noise levels ────────────────────────────
noise_levels = [0.05, 0.10, 0.20, 0.40]
noise_labels = ['5%', '10%', '20%', '40%']

print("=" * 70)
print("Lorenz-96 DA — FINAL VERSION")
print(f"Parameters: αA={ALPHA_A}, wb={WB}, da_interval={da_interval}")
print(f"Background perturbation: 5%")
print(f"Noise definition: fraction of signal (FIXED from noise/2 bug)")
print("=" * 70)

results = {}

for noise, label in zip(noise_levels, noise_labels):

    print(f"\nNoise level = {label}  (noise_level={noise})...")

    # FIXED: noise_level=noise (not noise/2)
    X_obs, obs_idx = generate_observations_96(
        X_true, noise_level=noise, obs_every=2
    )

    # Cell-NN DA — optimized parameters
    t0 = time.time()
    X_cnn, err_cnn = run_cellnn_lorenz96(
        X_true, X_obs, obs_idx,
        X0_bg, dt,
        da_interval=da_interval,
        da_on_periods=da_on_periods,
        alpha_A=ALPHA_A, wb=WB,
        K=K, F=F
    )
    time_cnn = time.time() - t0

    # 3D-Var DA
    t0 = time.time()
    X_3dv, err_3dv = run_3dvar_lorenz96(
        X_true, X_obs, obs_idx,
        X0_bg, dt,
        da_interval=da_interval,
        da_on_periods=da_on_periods,
        K=K, F=F
    )
    time_3dv = time.time() - t0

    # R-score — DA ON periods ONLY (correct definition)
    da_on_mask = np.zeros(N, dtype=bool)
    for start, end in da_on_periods:
        da_on_mask[start:end] = True

    r_cnn = np.mean((X_true[da_on_mask] - X_cnn[da_on_mask])**2)
    r_3dv = np.mean((X_true[da_on_mask] - X_3dv[da_on_mask])**2)

    cnn_errs = [e['mean_err'] for e in err_cnn]
    dv_errs  = [e['mean_err'] for e in err_3dv]

    results[label] = {
        'cnn_mean' : np.mean(cnn_errs),
        'dv_mean'  : np.mean(dv_errs),
        'r_cnn'    : r_cnn,
        'r_3dv'    : r_3dv,
        'X_cnn'    : X_cnn,
        'X_3dv'    : X_3dv,
        'err_cnn'  : err_cnn,
        'err_3dv'  : err_3dv,
        'time_cnn' : time_cnn,
        'time_3dv' : time_3dv,
    }

    print(f"  Cell-NN : mean_err={np.mean(cnn_errs):.4f}  R={r_cnn:.4f}  ({time_cnn:.1f}s)")
    print(f"  3D-Var  : mean_err={np.mean(dv_errs):.4f}  R={r_3dv:.4f}  ({time_3dv:.1f}s)")

# ── Results Table ─────────────────────────────────────
print(f"\n{'='*75}")
print(f"FINAL RESULTS TABLE (αA={ALPHA_A}, wb={WB})")
print(f"R-score = MSE during DA ON periods only")
print(f"{'='*75}")
print(f"{'Noise':<8} {'CNN_err':>10} {'3DV_err':>10} {'R_CNN':>10} {'R_3DV':>10} {'Winner':>10}")
print(f"{'-'*75}")
for label in noise_labels:
    r = results[label]
    winner = 'Cell-NN' if r['cnn_mean'] < r['dv_mean'] else '3D-Var'
    print(f"{label:<8} {r['cnn_mean']:>10.4f} {r['dv_mean']:>10.4f} "
          f"{r['r_cnn']:>10.4f} {r['r_3dv']:>10.4f} {winner:>10}")
print(f"{'='*75}")

# ── Save results ──────────────────────────────────────
noise_results = np.array([[results[l]['cnn_mean'],
                            results[l]['dv_mean'],
                            results[l]['r_cnn'],
                            results[l]['r_3dv']] for l in noise_labels])
np.save('results/lorenz96_noise_results_final.npy', noise_results)
print("\nResults saved → results/lorenz96_noise_results_final.npy")

# ── Plots ─────────────────────────────────────────────
fig, axes = plt.subplots(4, 2, figsize=(16, 18))

for idx, label in enumerate(noise_labels):
    r = results[label]

    # Left: X_1 trajectory
    axes[idx][0].plot(t, X_true[:, 0], 'b-', lw=0.8,
                      label='True', alpha=0.9)
    axes[idx][0].plot(t, r['X_3dv'][:, 0], 'r--', lw=0.8,
                      label=f'3D-Var err={r["dv_mean"]:.3f}')
    axes[idx][0].plot(t, r['X_cnn'][:, 0], color='purple',
                      lw=0.8, linestyle=':', 
                      label=f'Cell-NN err={r["cnn_mean"]:.3f}')
    axes[idx][0].axvspan(20, 35, alpha=0.1, color='gray',
                         label='DA OFF')
    axes[idx][0].set_title(f'Noise={label} — X_1(t)', fontsize=11)
    axes[idx][0].legend(fontsize=8)
    axes[idx][0].grid(True, alpha=0.3)

    # Right: DA error over time
    steps_cnn = [e['step']*dt for e in r['err_cnn']]
    steps_3dv = [e['step']*dt for e in r['err_3dv']]
    errs_cnn  = [e['mean_err'] for e in r['err_cnn']]
    errs_3dv  = [e['mean_err'] for e in r['err_3dv']]

    axes[idx][1].plot(steps_3dv, errs_3dv, 'r-', lw=0.8,
                      label=f'3D-Var R={r["r_3dv"]:.3f}')
    axes[idx][1].plot(steps_cnn, errs_cnn, color='purple',
                      lw=0.8, label=f'Cell-NN R={r["r_cnn"]:.3f}')
    axes[idx][1].set_title(f'Noise={label} — DA Error', fontsize=11)
    axes[idx][1].legend(fontsize=9)
    axes[idx][1].grid(True, alpha=0.3)
    axes[idx][1].set_xlabel('Time')
    axes[idx][1].set_ylabel('Mean |error|')

fig.suptitle(
    f'Lorenz-96 DA: Cell-NN vs 3D-Var — Different Noise Levels\n'
    f'αA={ALPHA_A}, wb={WB}, da_interval={da_interval}, perturb=5%\n'
    f'R-score = MSE during DA ON periods only',
    fontsize=13)
plt.tight_layout()
plt.savefig('results/lorenz96_noise_final.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved → results/lorenz96_noise_final.png")
