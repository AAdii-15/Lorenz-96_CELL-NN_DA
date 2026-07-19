"""
Template Parameter Sensitivity Analysis
=========================================
Testing different αA and wb values for
A^DA and B^DA templates to find optimal parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lorenz63  import (lorenz63, integrate_lorenz63,
                            generate_observations, SIGMA, BETA, RHO)
from src.lorenz96  import (integrate_lorenz96, make_initial_condition,
                            generate_observations_96)
from src.da_cellnn import run_da_cycle_cellnn
from src.lorenz96_da import run_cellnn_lorenz96

print("=" * 65)
print("Template Parameter Sensitivity Analysis")
print("=" * 65)

# ── Parameter ranges to test ────────────────────────
alpha_values = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
wb_values    = [1.0, 2.0, 3.0, 5.0, 10.0]

# ══════════════════════════════════════════════════
# PART 1: Lorenz-63 Sensitivity
# ══════════════════════════════════════════════════
print("\nPART 1: Lorenz-63 Parameter Sensitivity")
print("-" * 65)

w_true_ic     = np.array([1.0, 3.0, 5.0])
w_bg_ic       = np.array([1.1, 3.3, 5.5])
dt63          = 1e-3
N63           = 20000
t63, w_true63 = integrate_lorenz63(w_true_ic, (0, N63*dt63), dt63)
w_obs63       = generate_observations(w_true63, noise_level=0.05)
params63      = {'sigma': SIGMA, 'beta': BETA, 'rho': RHO}
da_on_63      = [(0, 7000), (15000, 20000)]

# Grid search
results63 = np.zeros((len(alpha_values), len(wb_values)))

print(f"{'αA':<8}", end="")
for wb in wb_values:
    print(f"{'wb='+str(wb):>10}", end="")
print()
print("-" * 58)

best_error63 = float('inf')
best_alpha63 = 0.1
best_wb63    = 3.0

for i, alpha in enumerate(alpha_values):
    print(f"{alpha:<8}", end="", flush=True)
    for j, wb in enumerate(wb_values):
        w_sim, errs = run_da_cycle_cellnn(
            w_true63, w_obs63, lorenz63, params63,
            dt63, w_bg_ic,
            da_interval=100,
            da_on_periods=da_on_63,
            alpha_A=alpha, wb=wb
        )
        me = np.mean([e['mean_err'] for e in errs]) if errs else 999
        results63[i, j] = me
        print(f"{me:>10.4f}", end="", flush=True)

        if me < best_error63:
            best_error63 = me
            best_alpha63 = alpha
            best_wb63    = wb
    print()

print(f"\nBest Lorenz-63: αA={best_alpha63}, wb={best_wb63}, "
      f"error={best_error63:.4f}")
print(f"Paper values:   αA=1.0,  wb=1.0,  "
      f"error={results63[alpha_values.index(1.0), wb_values.index(1.0)]:.4f}")

# ══════════════════════════════════════════════════
# PART 2: Lorenz-96 Sensitivity
# ══════════════════════════════════════════════════
print("\nPART 2: Lorenz-96 Parameter Sensitivity")
print("-" * 65)

X0_true96 = make_initial_condition(40, 8.0, perturb=0.0,  seed=42)
X0_bg96   = make_initial_condition(40, 8.0, perturb=0.05, seed=123)
dt96      = 1e-2
N96       = 3000  # shorter for speed
t96, X_true96 = integrate_lorenz96(X0_true96, (0, N96*dt96), dt96)
X_obs96, obs_idx = generate_observations_96(
    X_true96, noise_level=0.05, obs_every=2)
da_on_96  = [(0, 1500), (2000, 3000)]

results96 = np.zeros((len(alpha_values), len(wb_values)))

print(f"{'αA':<8}", end="")
for wb in wb_values:
    print(f"{'wb='+str(wb):>10}", end="")
print()
print("-" * 58)

best_error96 = float('inf')
best_alpha96 = 0.1
best_wb96    = 3.0

for i, alpha in enumerate(alpha_values):
    print(f"{alpha:<8}", end="", flush=True)
    for j, wb in enumerate(wb_values):
        X_sim, errs = run_cellnn_lorenz96(
            X_true96, X_obs96, obs_idx,
            X0_bg96, dt96,
            da_interval=25,
            da_on_periods=da_on_96,
            alpha_A=alpha, wb=wb,
            K=40, F=8.0
        )
        me = np.mean([e['mean_err'] for e in errs]) if errs else 999
        results96[i, j] = me
        print(f"{me:>10.4f}", end="", flush=True)

        if me < best_error96:
            best_error96 = me
            best_alpha96 = alpha
            best_wb96    = wb
    print()

print(f"\nBest Lorenz-96: αA={best_alpha96}, wb={best_wb96}, "
      f"error={best_error96:.4f}")
print(f"Paper values:   αA=1.0,  wb=1.0,  "
      f"error={results96[alpha_values.index(1.0), wb_values.index(1.0)]:.4f}")

# ══════════════════════════════════════════════════
# PART 3: Plots — Heatmaps
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Lorenz-63 heatmap
im1 = axes[0].imshow(results63, cmap='RdYlGn_r',
                      aspect='auto')
axes[0].set_xticks(range(len(wb_values)))
axes[0].set_xticklabels([str(w) for w in wb_values])
axes[0].set_yticks(range(len(alpha_values)))
axes[0].set_yticklabels([str(a) for a in alpha_values])
axes[0].set_xlabel('wb values', fontsize=11)
axes[0].set_ylabel('αA values', fontsize=11)
axes[0].set_title('(a) Lorenz-63: Mean Error\n(Green=better, Red=worse)',
                  fontsize=11)
plt.colorbar(im1, ax=axes[0])

# Add text annotations
for i in range(len(alpha_values)):
    for j in range(len(wb_values)):
        axes[0].text(j, i, f'{results63[i,j]:.3f}',
                    ha='center', va='center', fontsize=8)

# Mark best
bi = alpha_values.index(best_alpha63)
bj = wb_values.index(best_wb63)
axes[0].add_patch(plt.Rectangle((bj-0.5, bi-0.5), 1, 1,
                                  fill=False, color='blue', lw=3))
axes[0].text(bj, bi-0.4, 'BEST', ha='center',
             color='blue', fontsize=8, fontweight='bold')

# Mark paper values
pi = alpha_values.index(1.0)
pj = wb_values.index(1.0)
axes[0].add_patch(plt.Rectangle((pj-0.5, pi-0.5), 1, 1,
                                  fill=False, color='purple', lw=2,
                                  linestyle='--'))
axes[0].text(pj, pi+0.4, 'PAPER', ha='center',
             color='purple', fontsize=8)

# Lorenz-96 heatmap
im2 = axes[1].imshow(results96, cmap='RdYlGn_r',
                      aspect='auto')
axes[1].set_xticks(range(len(wb_values)))
axes[1].set_xticklabels([str(w) for w in wb_values])
axes[1].set_yticks(range(len(alpha_values)))
axes[1].set_yticklabels([str(a) for a in alpha_values])
axes[1].set_xlabel('wb values', fontsize=11)
axes[1].set_ylabel('αA values', fontsize=11)
axes[1].set_title('(b) Lorenz-96: Mean Error\n(Green=better, Red=worse)',
                  fontsize=11)
plt.colorbar(im2, ax=axes[1])

for i in range(len(alpha_values)):
    for j in range(len(wb_values)):
        axes[1].text(j, i, f'{results96[i,j]:.3f}',
                    ha='center', va='center', fontsize=8)

bi96 = alpha_values.index(best_alpha96)
bj96 = wb_values.index(best_wb96)
axes[1].add_patch(plt.Rectangle((bj96-0.5, bi96-0.5), 1, 1,
                                  fill=False, color='blue', lw=3))
axes[1].text(bj96, bi96-0.4, 'BEST', ha='center',
             color='blue', fontsize=8, fontweight='bold')

axes[1].add_patch(plt.Rectangle((pj-0.5, pi-0.5), 1, 1,
                                  fill=False, color='purple', lw=2,
                                  linestyle='--'))
axes[1].text(pj, pi+0.4, 'PAPER', ha='center',
             color='purple', fontsize=8)

plt.tight_layout()
plt.savefig('results/template_sensitivity.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved → results/template_sensitivity.png")

# ══════════════════════════════════════════════════
# PART 4: Summary
# ══════════════════════════════════════════════════
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"\nLorenz-63:")
print(f"  Paper values (αA=1.0, wb=1.0): "
      f"{results63[alpha_values.index(1.0), wb_values.index(1.0)]:.4f}")
print(f"  Best values  (αA={best_alpha63}, wb={best_wb63}): "
      f"{best_error63:.4f}")
improvement63 = (results63[alpha_values.index(1.0),
                            wb_values.index(1.0)] - best_error63)
print(f"  Improvement: {improvement63:.4f}")

print(f"\nLorenz-96:")
print(f"  Paper values (αA=1.0, wb=1.0): "
      f"{results96[alpha_values.index(1.0), wb_values.index(1.0)]:.4f}")
print(f"  Best values  (αA={best_alpha96}, wb={best_wb96}): "
      f"{best_error96:.4f}")
improvement96 = (results96[alpha_values.index(1.0),
                            wb_values.index(1.0)] - best_error96)
print(f"  Improvement: {improvement96:.4f}")
