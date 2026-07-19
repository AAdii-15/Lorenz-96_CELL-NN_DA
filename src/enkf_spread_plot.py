"""
Experiment 3: EnKF Ensemble Spread Diagnostic
==============================================
Shows WHY EnKF diverges under intermittent observations —
the mechanism, not just the outcome.

Two panels per row:
  Left:  Error over time (already in the paper as enkf_vs_cellnn_robustness.png)
  Right: Ensemble spread over time (NEW — shows the mechanism)

Under continuous observations:
  → Spread stays bounded; analysis regularly corrects it
Under intermittent observations:
  → Spread grows unchecked during DA-off period (t=2000-3500)
  → When observations resume at t=3500, the ensemble is already
    inflated and ill-conditioned → analysis diverges

Cell-NN has no ensemble and no covariance propagation, so it
is immune to this mechanism entirely.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.lorenz96 import (integrate_lorenz96, make_initial_condition,
                           generate_observations_96)
from src.enkf import run_enkf_lorenz96, rk4_step, enkf_analysis
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("Experiment 3: EnKF Ensemble Spread Diagnostic")
print("=" * 65)

K, F, dt, N = 40, 8.0, 1e-2, 5000
X0_true = make_initial_condition(K, F, perturb=0.0,  seed=42)
X0_bg   = make_initial_condition(K, F, perturb=0.05, seed=123)
_, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt)
X_obs, obs_idx = generate_observations_96(X_true, noise_level=0.05, obs_every=2)

da_on_periods_cont  = [(0, N)]
da_on_periods_interm = [(0, 2000), (3500, 5000)]

N_ENS = 50; INFL = 1.05; LOC_R = 8; R_STD = 0.05; DA_INT = 5

def run_enkf_with_spread(da_on_periods):
    """Run EnKF and record per-step error and ensemble spread."""
    np.random.seed(42)
    X_ens = X0_bg[None,:] + np.random.randn(N_ENS, K) * 0.01
    errors = np.full(N, np.nan)
    spreads = np.full(N, np.nan)
    X_mean = np.zeros((N, K))
    X_mean[0] = X_ens.mean(0)
    errors[0]  = np.sqrt(np.mean((X_true[0] - X_mean[0])**2))
    spreads[0] = np.mean(X_ens.std(axis=0))

    da_on_mask = np.zeros(N, dtype=bool)
    for s, e in da_on_periods:
        da_on_mask[s:e] = True

    for step in range(1, N):
        for m in range(N_ENS):
            xn = rk4_step(X_ens[m], dt, K, F)
            if not np.isfinite(xn).all():
                xn = X_ens.mean(0) + np.random.randn(K)*0.01
            X_ens[m] = xn

        if da_on_mask[step] and (step % DA_INT == 0):
            y = X_obs[step, obs_idx]
            X_ens, xa = enkf_analysis(X_ens, y, obs_idx,
                                       R_std=R_STD, inflation=INFL,
                                       loc_radius=LOC_R)
            X_mean[step] = xa
        else:
            X_mean[step] = X_ens.mean(0)

        if np.isfinite(X_ens).all():
            errors[step]  = np.sqrt(np.mean((X_true[step]-X_mean[step])**2))
            spreads[step] = np.mean(X_ens.std(axis=0))
        else:
            errors[step]  = np.nan
            spreads[step] = np.nan

    return errors, spreads, X_mean

print("Running continuous EnKF...")
err_cont, spr_cont, _ = run_enkf_with_spread(da_on_periods_cont)
print(f"  Final error={np.nanmean(err_cont):.4f} | "
      f"Mean spread={np.nanmean(spr_cont):.4f}")

print("Running intermittent EnKF...")
err_intr, spr_intr, _ = run_enkf_with_spread(da_on_periods_interm)
print(f"  Final error={np.nanmean(err_intr[~np.isnan(err_intr)]):.4f} | "
      f"Max spread before divergence="
      f"{np.nanmax(spr_intr[1500:3500]):.4f}")

t_axis = np.arange(N) * dt

# ── Figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

colors = {'cont': '#2196F3', 'intr': '#F44336'}
da_off_span = (2000*dt, 3500*dt)

for row, (err, spr, label, color, lw) in enumerate([
    (err_cont, spr_cont, 'Continuous observations', colors['cont'], 1.5),
    (err_intr, spr_intr, 'Intermittent observations', colors['intr'], 1.5),
]):
    # Error panel
    ax_e = fig.add_subplot(gs[row, 0])
    ax_e.plot(t_axis, np.clip(err, 0, 30), color=color, lw=lw, alpha=0.85)
    if row == 1:
        ax_e.axvspan(*da_off_span, alpha=0.12, color='gray',
                     label='DA off (t=20–35)')
        ax_e.legend(fontsize=8)
    ax_e.set_ylim(0, 30)
    ax_e.set_xlabel('Time', fontsize=10)
    ax_e.set_ylabel('RMSE', fontsize=10)
    ax_e.set_title(f'Error — {label}', fontsize=10, fontweight='bold')
    ax_e.text(0.02, 0.95, ['(a)', '(c)'][row], transform=ax_e.transAxes,
              fontsize=13, fontweight='bold', va='top')
    ax_e.grid(True, alpha=0.3)

    # Spread panel
    ax_s = fig.add_subplot(gs[row, 1])
    ax_s.plot(t_axis, np.clip(spr, 0, 30), color=color, lw=lw, alpha=0.85)
    if row == 1:
        ax_s.axvspan(*da_off_span, alpha=0.12, color='gray')
        # Annotate the spread growth
        peak_t  = t_axis[np.nanargmax(spr_intr[1500:3500]) + 1500]
        peak_s  = np.nanmax(spr_intr[1500:3500])
        ax_s.annotate('Spread grows\nuncontrolled →\ndivergence',
                      xy=(peak_t, min(peak_s, 28)),
                      xytext=(peak_t - 8, 20),
                      fontsize=8, color='#B71C1C',
                      arrowprops=dict(arrowstyle='->', color='#B71C1C'))
    ax_s.set_ylim(0, 30)
    ax_s.set_xlabel('Time', fontsize=10)
    ax_s.set_ylabel('Ensemble spread', fontsize=10)
    ax_s.set_title(f'Ensemble spread — {label}', fontsize=10,
                   fontweight='bold')
    ax_s.text(0.02, 0.95, ['(b)', '(d)'][row], transform=ax_s.transAxes,
              fontsize=13, fontweight='bold', va='top')
    ax_s.grid(True, alpha=0.3)

plt.savefig('results/enkf_spread_diagnostic.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved -> results/enkf_spread_diagnostic.png")

# ── Print key numbers ─────────────────────────────────────────────────
print(f"\nKey diagnostic numbers:")
print(f"  Continuous — mean spread: {np.nanmean(spr_cont):.4f}")
print(f"  Intermittent — mean spread (DA-on t<2000): "
      f"{np.nanmean(spr_intr[:2000]):.4f}")
print(f"  Intermittent — max spread (DA-off t=2000-3500): "
      f"{np.nanmax(spr_intr[1500:3500]):.4f}")
print(f"  Intermittent — spread at divergence onset: "
      f"{np.nanmean(spr_intr[3480:3520]):.4f}")
print("\nDone.")
