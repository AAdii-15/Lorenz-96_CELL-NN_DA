"""
Intermittency Gap-Length Experiment (E4)
==========================================
Tests Cell-NN and EnKF across multiple DA-off gap lengths:
  gap_steps ∈ {5, 10, 20, 40, 80}  (steps of da_interval=25 each)
  = physical gaps of ~1.25, 2.5, 5, 10, 20 Lorenz time units

For each gap length:
  - DA-ON period: steps 0-2000
  - DA-OFF gap:   inserted at step 2000 for N_gap steps
  - DA-ON resumes: until step 5000
  - Reports R-score for each method during DA-ON periods only

Shows where EnKF degradation begins and that Cell-NN stability
is not specific to one gap schedule.
Averaged over 5 seeds for statistical reliability.
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath('.')))
from src.lorenz96 import (integrate_lorenz96, make_initial_condition,
                           generate_observations_96)
from src.lorenz96_da import run_3dvar_lorenz96, run_cellnn_lorenz96
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("E4: Intermittency Gap-Length Experiment")
print("Tests Cell-NN and EnKF across 5 gap lengths, 5 seeds each")
print("=" * 70)

K, F, dt = 40, 8.0, 1e-2
N        = 5000
DA_INT   = 25

# Gap lengths in DA cycles (each = DA_INT * dt Lorenz time units)
GAP_CYCLES = [2, 5, 10, 20, 40]  # DA cycles of DA-off
GAP_STEPS  = [g * DA_INT for g in GAP_CYCLES]  # actual timesteps
GAP_LT     = [g * DA_INT * dt for g in GAP_CYCLES]  # Lorenz time units

SEEDS = [(42,123), (13,31), (99,17), (55,88), (101,202)]

print(f"\nGap lengths tested:")
for gc, gs, lt in zip(GAP_CYCLES, GAP_STEPS, GAP_LT):
    print(f"  {gc:3d} DA cycles = {gs:5d} steps = {lt:.1f} Lorenz time units")

def build_da_periods(n_total, gap_start, gap_steps):
    """DA-on for 0→gap_start, DA-off for gap_steps, then DA-on again."""
    end_gap = min(gap_start + gap_steps, n_total)
    periods = []
    if gap_start > 0:
        periods.append((0, gap_start))
    if end_gap < n_total:
        periods.append((end_gap, n_total))
    return periods

def compute_r_score(X_true, X_pred, da_on_periods, N):
    """MSE during DA-on periods only."""
    mask = np.zeros(N, dtype=bool)
    for s, e in da_on_periods:
        mask[s:e] = True
    if mask.sum() == 0:
        return np.nan
    return np.mean((X_true[mask] - X_pred[mask])**2)

# Also run no-gap (continuous DA) as baseline
GAP_LABEL = ['No gap\n(continuous)'] + \
            [f'{gc} cycles\n({lt:.1f} LTU)' 
             for gc, lt in zip(GAP_CYCLES, GAP_LT)]

all_gap_cycles = [0] + GAP_CYCLES
all_gap_steps  = [0] + GAP_STEPS
GAP_START      = 2000  # where the gap begins

results = {gc: {'3dvar': [], 'cellnn': []} 
           for gc in all_gap_cycles}

print(f"\n{'Gap':>10}  {'Seed':>10}  {'3D-Var R':>10}  "
      f"{'Cell-NN R':>10}  {'CeNN better?':>13}")
print("-" * 60)

for gc, gs in zip(all_gap_cycles, all_gap_steps):
    for true_seed, bg_seed in SEEDS:
        X0_true = make_initial_condition(K, F, perturb=0.0,  seed=true_seed)
        X0_bg   = make_initial_condition(K, F, perturb=0.05, seed=bg_seed)
        _, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt)
        np.random.seed(true_seed)
        X_obs, obs_idx = generate_observations_96(
            X_true, noise_level=0.05, obs_every=2)

        da_on_periods = build_da_periods(N, GAP_START, gs)

        X_3dvar, _ = run_3dvar_lorenz96(
            X_true, X_obs, obs_idx, X0_bg, dt,
            da_interval=DA_INT, da_on_periods=da_on_periods, K=K, F=F)

        X_cellnn, _ = run_cellnn_lorenz96(
            X_true, X_obs, obs_idx, X0_bg, dt,
            da_interval=DA_INT, da_on_periods=da_on_periods,
            alpha_A=1.0, wb=1.0, K=K, F=F)

        r_3d  = compute_r_score(X_true, X_3dvar,  da_on_periods, N)
        r_cnn = compute_r_score(X_true, X_cellnn, da_on_periods, N)

        results[gc]['3dvar'].append(r_3d)
        results[gc]['cellnn'].append(r_cnn)

        better = "✓" if r_cnn < r_3d else "✗"
        seed_label = f"({true_seed},{bg_seed})"
        print(f"  gap={gc:3d}  {seed_label:>10}  "
              f"{r_3d:>10.4f}  {r_cnn:>10.4f}  {better:>13}")

# ── Summary table ────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("SUMMARY: Mean R-score across 5 seeds (lower is better)")
print(f"{'='*70}")
print(f"{'Gap':>20}  {'3D-Var':>10}  {'Cell-NN':>10}  "
      f"{'CeNN improv':>12}  {'CeNN stable?':>12}")
print("-" * 70)

gap_labels_short = ['No gap (0)'] + \
    [f'{gc} cycles ({lt:.0f} LTU)' 
     for gc, lt in zip(GAP_CYCLES, GAP_LT)]

summary_rows = []
for gc, label in zip(all_gap_cycles, gap_labels_short):
    r3   = np.array(results[gc]['3dvar'])
    rc   = np.array(results[gc]['cellnn'])
    r3_m = r3.mean(); rc_m = rc.mean()
    imp  = (r3_m - rc_m) / r3_m * 100
    cnn_stable = "✓" if all(r < 5.0 for r in rc) else "✗ UNSTABLE"
    summary_rows.append((gc, label, r3_m, rc_m, imp, rc.max(), r3.max()))
    print(f"  {label:<22}{r3_m:>10.4f}  {rc_m:>10.4f}  "
          f"{imp:>11.1f}%  {cnn_stable:>12}")

print(f"\nNote: R-score is MSE during DA-ON periods.")
print(f"      Stable = max R-score across seeds < 5.0")
print(f"{'='*70}")

# ── Key finding ──────────────────────────────────────────────────────────
print("\nKEY FINDING:")
for gc, label, r3_m, rc_m, imp, rc_max, r3_max in summary_rows:
    if gc == 0:
        print(f"  No gap:  3D-Var={r3_m:.4f}, Cell-NN={rc_m:.4f}")
    else:
        r3_deg = ((r3_m / summary_rows[0][2]) - 1) * 100
        rc_deg = ((rc_m / summary_rows[0][3]) - 1) * 100
        print(f"  Gap={gc:2d}c: 3D-Var degrades {r3_deg:+.1f}%, "
              f"Cell-NN degrades {rc_deg:+.1f}%  "
              f"[3DVar_max={r3_max:.2f}, CeNN_max={rc_max:.2f}]")

np.savez('results/intermittency_gap_length.npz',
         gap_cycles=np.array(all_gap_cycles),
         gap_steps=np.array(all_gap_steps),
         results_3dvar=np.array([results[gc]['3dvar'] 
                                  for gc in all_gap_cycles]),
         results_cellnn=np.array([results[gc]['cellnn'] 
                                   for gc in all_gap_cycles]))
print("\nSaved -> results/intermittency_gap_length.npz")
print("\nDone.")
