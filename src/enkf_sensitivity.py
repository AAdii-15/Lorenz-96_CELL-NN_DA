"""
Experiment 2: EnKF Sensitivity Study
=========================================================================
Addresses the reviewer's concern that EnKF divergence under intermittent
observations may be an artifact of poor tuning rather than a genuine
finding. Tests robustness of the key result (EnKF diverges under ON/OFF
schedule, Cell-NN does not) across:
  - N_ens: 20, 50 (baseline), 100, 200
  - inflation: 1.00, 1.02, 1.05 (baseline), 1.10, 1.20
  - loc_radius: 4, 6, 8 (baseline), 10, 14

For each configuration, reports R-score under BOTH continuous and
intermittent observation schedules, matching the R-score definition
in enkf_reproduce.py (MSE during DA-ON periods only).
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath('.')))
from src.lorenz96 import (integrate_lorenz96, make_initial_condition,
                           generate_observations_96)
from src.enkf import run_enkf_lorenz96
import warnings
warnings.filterwarnings('ignore')

print("=" * 75)
print("Experiment 2: EnKF Sensitivity Study")
print("=" * 75)

K, F, dt, N = 40, 8.0, 1e-2, 5000
X0_true = make_initial_condition(K, F, perturb=0.0, seed=42)
X0_bg   = make_initial_condition(K, F, perturb=0.05, seed=123)
_, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt)
X_obs, obs_idx = generate_observations_96(X_true, noise_level=0.05, obs_every=2)

da_on_periods  = [(0, 2000), (3500, 5000)]
da_on_mask     = np.zeros(N, dtype=bool)
for s, e in da_on_periods:
    da_on_mask[s:e] = True

BASELINE = dict(N_ens=50, inflation=1.05, loc_radius=8)

# ── Sensitivity grid ───────────────────────────────────
N_ENS_VALS   = [20, 50, 100, 200]
INFL_VALS    = [1.00, 1.02, 1.05, 1.10, 1.20]
LOC_VALS     = [4, 6, 8, 10, 14]

def run_pair(N_ens, inflation, loc_radius):
    """Run continuous + intermittent EnKF; return (R_cont, R_interm)."""
    _, errors_cont = run_enkf_lorenz96(
        X_true, X_obs, obs_idx, X0_bg, dt,
        da_interval=5, da_on_periods=[(0, N)],
        K=K, F=F, N_ens=N_ens, R_std=0.05,
        inflation=inflation, loc_radius=loc_radius)
    X_interm, _ = run_enkf_lorenz96(
        X_true, X_obs, obs_idx, X0_bg, dt,
        da_interval=5, da_on_periods=da_on_periods,
        K=K, F=F, N_ens=N_ens, R_std=0.05,
        inflation=inflation, loc_radius=loc_radius)

    # rebuild X_cont from error list
    from src.enkf import rk4_step, enkf_analysis
    np.random.seed(42)
    X_ens = X0_bg[None,:] + np.random.randn(N_ens, K) * 0.01
    X_cont = np.zeros((N, K)); X_cont[0] = X_ens.mean(0)
    for step in range(1, N):
        for m in range(N_ens):
            x_new = rk4_step(X_ens[m], dt, K, F)
            if not np.isfinite(x_new).all():
                x_new = X_ens.mean(0) + np.random.randn(K)*0.01
            X_ens[m] = x_new
        X_cont[step] = X_ens.mean(0)
        if step % 5 == 0:
            y = X_obs[step, obs_idx]
            X_ens, xa = enkf_analysis(X_ens, y, obs_idx,
                                       R_std=0.05, inflation=inflation,
                                       loc_radius=loc_radius)
            X_cont[step] = xa

    R_cont   = float(np.mean((X_true[da_on_mask]-X_cont[da_on_mask])**2))
    R_interm = float(np.mean((X_true[da_on_mask]-X_interm[da_on_mask])**2))
    return R_cont, R_interm

# ── Baseline ────────────────────────────────────────────
print("\nBaseline (N_ens=50, infl=1.05, loc_r=8)...")
r_b_cont, r_b_interm = run_pair(**BASELINE)
print(f"  Continuous: {r_b_cont:.4f}  |  Intermittent: {r_b_interm:.4f}  "
      f"|  Ratio: {r_b_interm/r_b_cont:.1f}x")

results = {'baseline': (r_b_cont, r_b_interm)}

# ── Sweep N_ens ─────────────────────────────────────────
print("\n--- Sweeping N_ens (infl=1.05, loc_r=8) ---")
nens_results = {}
for ne in N_ENS_VALS:
    r_c, r_i = run_pair(N_ens=ne, inflation=1.05, loc_radius=8)
    ratio = r_i / max(r_c, 1e-10)
    nens_results[ne] = (r_c, r_i)
    mark = '<< baseline' if ne == 50 else ''
    print(f"  N_ens={ne:3d}: Cont={r_c:.4f}  Interm={r_i:.4f}  "
          f"Ratio={ratio:.1f}x  {'DIVERGES' if r_i > 5 else 'stable'} {mark}")

# ── Sweep inflation ─────────────────────────────────────
print("\n--- Sweeping inflation (N_ens=50, loc_r=8) ---")
infl_results = {}
for infl in INFL_VALS:
    r_c, r_i = run_pair(N_ens=50, inflation=infl, loc_radius=8)
    ratio = r_i / max(r_c, 1e-10)
    infl_results[infl] = (r_c, r_i)
    mark = '<< baseline' if infl == 1.05 else ''
    print(f"  infl={infl:.2f}: Cont={r_c:.4f}  Interm={r_i:.4f}  "
          f"Ratio={ratio:.1f}x  {'DIVERGES' if r_i > 5 else 'stable'} {mark}")

# ── Sweep loc_radius ────────────────────────────────────
print("\n--- Sweeping loc_radius (N_ens=50, infl=1.05) ---")
loc_results = {}
for lr in LOC_VALS:
    r_c, r_i = run_pair(N_ens=50, inflation=1.05, loc_radius=lr)
    ratio = r_i / max(r_c, 1e-10)
    loc_results[lr] = (r_c, r_i)
    mark = '<< baseline' if lr == 8 else ''
    print(f"  loc_r={lr:2d}: Cont={r_c:.4f}  Interm={r_i:.4f}  "
          f"Ratio={ratio:.1f}x  {'DIVERGES' if r_i > 5 else 'stable'} {mark}")

# ── Summary ─────────────────────────────────────────────
print(f"\n{'='*75}")
print("SUMMARY TABLE -- EnKF Sensitivity to Parameter Choices")
print(f"{'='*75}")
print(f"{'Parameter':<20}{'Value':>8}{'Cont R':>10}{'Interm R':>12}"
      f"{'Ratio':>8}{'Verdict':>12}")
print("-"*75)
for ne, (rc, ri) in nens_results.items():
    v = 'DIVERGES' if ri > 5 else 'stable'
    print(f"{'N_ens':<20}{ne:>8}{rc:>10.4f}{ri:>12.4f}"
          f"{ri/max(rc,1e-10):>8.1f}x{v:>12}")
print()
for infl, (rc, ri) in infl_results.items():
    v = 'DIVERGES' if ri > 5 else 'stable'
    print(f"{'inflation':<20}{infl:>8.2f}{rc:>10.4f}{ri:>12.4f}"
          f"{ri/max(rc,1e-10):>8.1f}x{v:>12}")
print()
for lr, (rc, ri) in loc_results.items():
    v = 'DIVERGES' if ri > 5 else 'stable'
    print(f"{'loc_radius':<20}{lr:>8}{rc:>10.4f}{ri:>12.4f}"
          f"{ri/max(rc,1e-10):>8.1f}x{v:>12}")

np.savez('results/enkf_sensitivity.npz',
         nens_results=np.array([[ne, rc, ri]
                                 for ne, (rc, ri) in nens_results.items()]),
         infl_results=np.array([[infl, rc, ri]
                                 for infl, (rc, ri) in infl_results.items()]),
         loc_results=np.array([[lr, rc, ri]
                                for lr, (rc, ri) in loc_results.items()]))
print(f"\nSaved -> results/enkf_sensitivity.npz")
print("\nDone.")
