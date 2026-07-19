"""
Lorenz-96: 20-seed statistical validation with Wilcoxon signed-rank test
=========================================================================
Expands the 5-seed CI from the paper to 20 independent realizations.
Uses Wilcoxon signed-rank test (non-parametric, paired) since normality
cannot be assumed at n=20. Same exact parameters as the paper throughout.
Seeds 0-4 are identical to the existing 5-seed results for continuity.
"""
import numpy as np
from scipy import stats
import sys, os
sys.path.append(os.path.dirname(os.path.abspath('.')))
from src.lorenz96 import (integrate_lorenz96, make_initial_condition,
                           generate_observations_96)
from src.lorenz96_da import run_3dvar_lorenz96, run_cellnn_lorenz96
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Lorenz-96: 20-Seed Statistical Validation + Wilcoxon Test")
print("Parameters: da_interval=25, alpha_A=1.0, noise=0.05, obs_every=2")
print("=" * 70)

K, F, dt, N = 40, 8.0, 1e-2, 5000

# Original 5 seeds first (indices 0-4), then 15 new ones
# Each tuple: (true_IC_seed, background_IC_seed)
SEEDS = [
    # --- Original 5 (must match paper exactly) ---
    (42,  123),   # seed 0 = original paper primary seed
    (7,    77),
    (13,   31),
    (99,   17),
    (55,   88),
    # --- 15 new seeds ---
    (101, 202),
    (303, 404),
    (505, 606),
    (707, 808),
    (11,   22),
    (33,   44),
    (66,   77),
    (88,   99),
    (111, 333),
    (222, 444),
    (555, 666),
    (777, 888),
    (123, 456),
    (789, 321),
    (147, 258),
]
assert len(SEEDS) == 20, "Must have exactly 20 seeds"

da_on_periods = [(0, 2000), (3500, 5000)]
da_on_mask    = np.zeros(N, dtype=bool)
for s, e in da_on_periods:
    da_on_mask[s:e] = True

results_3dvar, results_cellnn = [], []

print(f"\n{'Seed':>12}  {'3D-Var':>10}  {'Cell-NN':>10}  "
      f"{'Improv%':>10}  {'Better?':>8}")
print("-" * 60)

for i, (ts, bs) in enumerate(SEEDS):
    X0_true = make_initial_condition(K, F, perturb=0.0,  seed=ts)
    X0_bg   = make_initial_condition(K, F, perturb=0.05, seed=bs)
    _, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt)

    np.random.seed(ts)
    X_obs, obs_idx = generate_observations_96(
        X_true, noise_level=0.05, obs_every=2)

    X_3dvar, _ = run_3dvar_lorenz96(
        X_true, X_obs, obs_idx, X0_bg, dt,
        da_interval=25, da_on_periods=da_on_periods, K=K, F=F)

    X_cellnn, _ = run_cellnn_lorenz96(
        X_true, X_obs, obs_idx, X0_bg, dt,
        da_interval=25, da_on_periods=da_on_periods,
        alpha_A=1.0, wb=1.0, K=K, F=F)

    err_3d  = np.sqrt(np.mean(
        (X_true[da_on_mask] - X_3dvar[da_on_mask])**2))
    err_cnn = np.sqrt(np.mean(
        (X_true[da_on_mask] - X_cellnn[da_on_mask])**2))
    improv  = (err_3d - err_cnn) / err_3d * 100
    better  = "✓" if improv > 0 else "✗"
    label   = "(paper)" if i == 0 else ""

    results_3dvar.append(err_3d)
    results_cellnn.append(err_cnn)

    print(f"  {ts:3d},{bs:3d}  {err_3d:>10.4f}  {err_cnn:>10.4f}  "
          f"{improv:>9.1f}%  {better:>8} {label}")

r3  = np.array(results_3dvar)
rc  = np.array(results_cellnn)
imp = (r3 - rc) / r3 * 100
diffs = r3 - rc  # positive = Cell-NN better

# ── Wilcoxon signed-rank test ─────────────────────────────────────────
# H0: median difference = 0
# H1: Cell-NN RMSE is systematically lower than 3D-Var RMSE
stat, p_two = stats.wilcoxon(r3, rc, alternative='two-sided')
stat, p_one = stats.wilcoxon(r3, rc, alternative='greater')

# Effect size: matched-pairs r = Z / sqrt(N)
# Approximate Z from p-value for reporting
from scipy.stats import norm as sp_norm
z_approx = sp_norm.ppf(1 - p_one)
effect_r  = z_approx / np.sqrt(len(SEEDS))

# Bootstrap 95% CI on mean improvement
np.random.seed(42)
boot_imps = [
    np.mean((r3[idx] - rc[idx]) / r3[idx] * 100)
    for _ in range(10000)
    for idx in [np.random.choice(len(SEEDS), len(SEEDS), replace=True)]
]
ci_lo, ci_hi = np.percentile(boot_imps, [2.5, 97.5])

print(f"\n{'='*70}")
print("20-SEED STATISTICAL RESULTS")
print(f"{'='*70}")
print(f"\n{'Metric':<35} {'3D-Var':>10} {'Cell-NN':>10}")
print("-" * 57)
print(f"{'Mean RMSE (DA-on periods)':<35} {r3.mean():>10.4f} {rc.mean():>10.4f}")
print(f"{'Std RMSE':<35} {r3.std():>10.4f}  {rc.std():>10.4f}")
print(f"{'Min RMSE':<35} {r3.min():>10.4f}  {rc.min():>10.4f}")
print(f"{'Max RMSE':<35} {r3.max():>10.4f}  {rc.max():>10.4f}")

print(f"\n{'Improvement (Cell-NN over 3D-Var)':}")
print(f"  Mean improvement:   {imp.mean():.1f}%")
print(f"  Std improvement:    {imp.std():.1f}%")
print(f"  Range:              {imp.min():.1f}% to {imp.max():.1f}%")
print(f"  Bootstrap 95% CI:   [{ci_lo:.1f}%, {ci_hi:.1f}%]")
print(f"  Positive in:        {(imp>0).sum()}/{len(SEEDS)} seeds")

print(f"\n{'Wilcoxon Signed-Rank Test (paired, non-parametric)':}")
print(f"  H0: median(RMSE_3dvar - RMSE_cellnn) = 0")
print(f"  Statistic W:        {stat:.1f}")
print(f"  p-value (two-sided):{p_two:.4f}")
print(f"  p-value (one-sided):{p_one:.4f}")
print(f"  Effect size r:      {effect_r:.3f}")
sig = "YES" if p_one < 0.05 else "NO"
print(f"  Significant (p<0.05, one-sided): {sig}")
print(f"{'='*70}")

# ── Sanity check: original 5 seeds must match prior results ──────────
print("\nSanity check (original 5 seeds, should match prior run):")
for i in range(5):
    imp_i = (r3[i]-rc[i])/r3[i]*100
    print(f"  seed {SEEDS[i]}: 3D-Var={r3[i]:.4f}, "
          f"Cell-NN={rc[i]:.4f}, Improv={imp_i:.1f}%")

np.savez('results/lorenz96_20seed_wilcoxon.npz',
         seeds=np.array(SEEDS), r3dvar=r3, rcellnn=rc,
         improvement=imp, ci=np.array([ci_lo, ci_hi]),
         wilcoxon_stat=stat, p_two=p_two, p_one=p_one,
         effect_r=effect_r)
print("\nSaved -> results/lorenz96_20seed_wilcoxon.npz")
print("\nDone.")
