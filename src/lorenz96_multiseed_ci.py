"""
Lorenz-96 Multi-Seed Confidence Intervals -- CORRECTED
=======================================================
Uses EXACT parameters from lorenz96_da_stable.py (the script
that produced the 0.3435/0.3103 results in the paper):
  da_interval = 25  (NOT 5 -- was the bug in prior attempt)
  alpha_A=1.0, wb=1.0
  noise_level=0.05
  obs_every=2
  perturb=0.05

5 independent (IC seed, obs seed) pairs.
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath('.')))
from src.lorenz96 import (integrate_lorenz96, make_initial_condition,
                           generate_observations_96)
from src.lorenz96_da import run_3dvar_lorenz96, run_cellnn_lorenz96
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("Experiment C: Lorenz-96 Multi-Seed CI (5 realizations)")
print("EXACT parameters: da_interval=25, alpha_A=1.0, wb=1.0")
print("noise_level=0.05, obs_every=2, perturb=0.05")
print("=" * 65)

K, F, dt, N = 40, 8.0, 1e-2, 5000
DA_INTERVAL   = 25     # EXACT match to original paper
ALPHA_A, WB   = 1.0, 1.0
NOISE_LEVEL   = 0.05
OBS_EVERY     = 2

da_on_periods = [(0, 2000), (3500, 5000)]
da_on_mask    = np.zeros(N, dtype=bool)
for s, e in da_on_periods:
    da_on_mask[s:e] = True

# Seed 0 = original paper seed (sanity check: should reproduce 0.3435/0.3103)
SEEDS = [
    (42,  123),   # seed 0 = original paper -- should match
    (7,    77),
    (13,   31),
    (99,   17),
    (55,   88),
]

results_3dvar, results_cellnn = [], []

for i, (true_seed, bg_seed) in enumerate(SEEDS):
    label = "(original paper seed)" if i == 0 else ""
    print(f"\n  seed=({true_seed},{bg_seed}) {label}...")

    X0_true = make_initial_condition(K, F, perturb=0.0,  seed=true_seed)
    X0_bg   = make_initial_condition(K, F, perturb=0.05, seed=bg_seed)
    _, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt)

    np.random.seed(true_seed)
    X_obs, obs_idx = generate_observations_96(
        X_true, noise_level=NOISE_LEVEL, obs_every=OBS_EVERY)

    X_3dvar, _ = run_3dvar_lorenz96(
        X_true, X_obs, obs_idx, X0_bg, dt,
        da_interval=DA_INTERVAL,
        da_on_periods=da_on_periods,
        K=K, F=F)

    X_cellnn, _ = run_cellnn_lorenz96(
        X_true, X_obs, obs_idx, X0_bg, dt,
        da_interval=DA_INTERVAL,
        da_on_periods=da_on_periods,
        alpha_A=ALPHA_A, wb=WB, K=K, F=F)

    err_3dvar  = np.sqrt(np.mean(
        (X_true[da_on_mask] - X_3dvar[da_on_mask])**2))
    err_cellnn = np.sqrt(np.mean(
        (X_true[da_on_mask] - X_cellnn[da_on_mask])**2))
    improv = (err_3dvar - err_cellnn) / err_3dvar * 100

    results_3dvar.append(err_3dvar)
    results_cellnn.append(err_cellnn)
    print(f"    3D-Var={err_3dvar:.4f}, Cell-NN={err_cellnn:.4f}, "
          f"Improv={improv:.1f}%")

r3 = np.array(results_3dvar)
rc = np.array(results_cellnn)
improv_arr = (r3 - rc) / r3 * 100

print(f"\n{'='*65}")
print("MULTI-SEED CI RESULTS")
print(f"{'='*65}")
print(f"Sanity check (seed 0 should match paper 0.3435 / 0.3103):")
print(f"  3D-Var[0] = {r3[0]:.4f}, Cell-NN[0] = {rc[0]:.4f}")
print()
print(f"{'Method':<20}{'Mean':>10}{'Std':>8}{'Min':>8}{'Max':>8}")
print("-"*55)
print(f"{'3D-Var':<20}{r3.mean():>10.4f}{r3.std():>8.4f}"
      f"{r3.min():>8.4f}{r3.max():>8.4f}")
print(f"{'Cell-NN':<20}{rc.mean():>10.4f}{rc.std():>8.4f}"
      f"{rc.min():>8.4f}{rc.max():>8.4f}")
print(f"{'Improvement%':<20}{improv_arr.mean():>9.1f}%"
      f"{improv_arr.std():>7.1f}%"
      f"{improv_arr.min():>7.1f}%"
      f"{improv_arr.max():>7.1f}%")
print(f"{'='*65}")
print(f"\nCell-NN improves on 3D-Var by "
      f"{improv_arr.mean():.1f}% ± {improv_arr.std():.1f}% "
      f"across 5 independent realizations "
      f"(range {improv_arr.min():.1f}%–{improv_arr.max():.1f}%).")
print(f"Positive in all 5 seeds: {all(i > 0 for i in improv_arr)}")
print(f"Positive in {sum(i > 0 for i in improv_arr)}/5 seeds.")

np.savez('results/lorenz96_multiseed_ci.npz',
         results_3dvar=r3, results_cellnn=rc,
         improv=improv_arr, seeds=np.array(SEEDS))
print("\nSaved -> results/lorenz96_multiseed_ci.npz")
print("\nDone.")
