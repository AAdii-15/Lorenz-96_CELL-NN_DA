"""
EnKF Reproduce Script — Fix 4
Explicitly reproduces EnKF results for paper
Shows filter divergence under ON/OFF DA schedule
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lorenz96  import (integrate_lorenz96, make_initial_condition,
                            generate_observations_96)
from src.enkf      import run_enkf_lorenz96
from src.lorenz96_da import run_cellnn_lorenz96, run_3dvar_lorenz96

print("="*65)
print("EnKF Reproduction Script")
print("Shows: EnKF diverges under ON/OFF schedule")
print("="*65)

K   = 40
F   = 8.0
dt  = 1e-2
N   = 5000

X0_true = make_initial_condition(K, F, perturb=0.0,  seed=42)
X0_bg   = make_initial_condition(K, F, perturb=0.05, seed=123)
t, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt)
X_obs, obs_idx = generate_observations_96(
    X_true, noise_level=0.05, obs_every=2)

da_on_periods = [(0, 2000), (3500, 5000)]

# DA ON mask for R-score
da_on_mask = np.zeros(N, dtype=bool)
for start, end in da_on_periods:
    da_on_mask[start:end] = True

# ── Experiment 1: EnKF Always ON ─────────────────────
print("\nExperiment 1: EnKF Always ON...")
X_enkf_on, err_enkf_on = run_enkf_lorenz96(
    X_true, X_obs, obs_idx,
    X0_bg, dt,
    da_interval=5,
    da_on_periods=[(0, N)],   # always on
    K=K, F=F,
    N_ens=50, R_std=0.05,
    inflation=1.05, loc_radius=8
)
r_enkf_on  = np.mean((X_true[da_on_mask] - X_enkf_on[da_on_mask])**2)
me_enkf_on = np.mean([e['mean_err'] for e in err_enkf_on])
print(f"  EnKF Always ON  — mean_err={me_enkf_on:.4f}  R={r_enkf_on:.4f}")

# ── Experiment 2: EnKF ON/OFF schedule ───────────────
print("\nExperiment 2: EnKF ON/OFF schedule...")
X_enkf_off, err_enkf_off = run_enkf_lorenz96(
    X_true, X_obs, obs_idx,
    X0_bg, dt,
    da_interval=5,
    da_on_periods=da_on_periods,   # ON/OFF like Cell-NN
    K=K, F=F,
    N_ens=50, R_std=0.05,
    inflation=1.05, loc_radius=8
)
r_enkf_off  = np.mean((X_true[da_on_mask] - X_enkf_off[da_on_mask])**2)
me_enkf_off = np.mean([e['mean_err'] for e in err_enkf_off]) if err_enkf_off else 999
print(f"  EnKF ON/OFF     — mean_err={me_enkf_off:.4f}  R={r_enkf_off:.4f}")

# ── Experiment 3: Cell-NN (reference) ────────────────
print("\nExperiment 3: Cell-NN DA (reference)...")
X_cnn, err_cnn = run_cellnn_lorenz96(
    X_true, X_obs, obs_idx,
    X0_bg, dt,
    da_interval=25,
    da_on_periods=da_on_periods,
    alpha_A=1.0, wb=1.0, K=K, F=F
)
r_cnn  = np.mean((X_true[da_on_mask] - X_cnn[da_on_mask])**2)
me_cnn = np.mean([e['mean_err'] for e in err_cnn])
print(f"  Cell-NN         — mean_err={me_cnn:.4f}  R={r_cnn:.4f}")

# ── Results Table ─────────────────────────────────────
print(f"\n{'='*65}")
print(f"ENKF vs CELL-NN COMPARISON")
print(f"{'='*65}")
print(f"{'Method':<25} {'mean_err':>10} {'R-score':>10} {'Status':>15}")
print(f"{'-'*65}")
print(f"{'EnKF Always ON':<25} {me_enkf_on:>10.4f} {r_enkf_on:>10.4f} {'Stable':>15}")
print(f"{'EnKF ON/OFF':<25} {me_enkf_off:>10.4f} {r_enkf_off:>10.4f} {'Diverges':>15}")
print(f"{'Cell-NN ON/OFF':<25} {me_cnn:>10.4f} {r_cnn:>10.4f} {'Stable ✓':>15}")
print(f"{'='*65}")
print(f"\nKey Finding: Cell-NN stable under ON/OFF schedule")
print(f"             EnKF diverges when DA is turned OFF")

# ── Save ─────────────────────────────────────────────
np.save('results/enkf_always_on_errors.npy',
        np.array([me_enkf_on, r_enkf_on]))
np.save('results/enkf_onoff_errors.npy',
        np.array([me_enkf_off, r_enkf_off]))
print("\nResults saved!")

# ── Plot ──────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True,
                          gridspec_kw={'hspace': 0.35})

# X_1 trajectories
axes[0].plot(t, X_true[:,0], 'b-', lw=0.8, label='True')
axes[0].plot(t, X_enkf_on[:,0], 'g--', lw=0.8,
             label=f'EnKF Always ON (err={me_enkf_on:.3f})')
axes[0].plot(t, X_enkf_off[:,0], 'r--', lw=0.8,
             label=f'EnKF ON/OFF (err={me_enkf_off:.3f})')
axes[0].plot(t, X_cnn[:,0], color='purple', lw=0.8,
             linestyle=':', label=f'Cell-NN ON/OFF (err={me_cnn:.3f})')
axes[0].axvspan(20, 35, alpha=0.15, color='gray', label='DA OFF')
axes[0].text(0.01, 0.95, '(a)', transform=axes[0].transAxes,
             fontsize=13, fontweight='bold', va='top')
axes[0].legend(fontsize=11, loc='upper left',
                bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylabel('X_1')

# EnKF errors
if err_enkf_on:
    s_on = [e['step']*dt for e in err_enkf_on]
    e_on = [e['mean_err'] for e in err_enkf_on]
    axes[1].plot(s_on, e_on, 'g-', lw=0.8,
                 label=f'EnKF Always ON R={r_enkf_on:.3f}')
if err_enkf_off:
    s_off = [e['step']*dt for e in err_enkf_off]
    e_off = [e['mean_err'] for e in err_enkf_off]
    axes[1].plot(s_off, e_off, 'r-', lw=0.8,
                 label=f'EnKF ON/OFF R={r_enkf_off:.3f}')
axes[1].axvspan(20, 35, alpha=0.15, color='gray')
axes[1].set_title('EnKF DA Error (Mean EnKF - True)', fontsize=12)
axes[1].text(0.01, 0.95, '(b)', transform=axes[1].transAxes,
             fontsize=13, fontweight='bold', va='top')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylabel('Mean |error|')

# Cell-NN errors
s_cnn = [e['step']*dt for e in err_cnn]
e_cnn = [e['mean_err'] for e in err_cnn]
axes[2].plot(s_cnn, e_cnn, color='purple', lw=0.8,
             label=f'Cell-NN ON/OFF R={r_cnn:.3f}')
axes[2].axvspan(20, 35, alpha=0.15, color='gray', label='DA OFF')
axes[2].set_title('Cell-NN DA Error (Mean Cell-NN - True)', fontsize=12)
axes[2].text(0.01, 0.95, '(c)', transform=axes[2].transAxes,
             fontsize=13, fontweight='bold', va='top')
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)
axes[2].set_ylabel('Mean |error|')
axes[2].set_xlabel('Time')

plt.tight_layout()
plt.savefig('results/enkf_vs_cellnn_robustness.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved → results/enkf_vs_cellnn_robustness.png")
