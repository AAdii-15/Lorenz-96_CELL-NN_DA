"""
Experiment 2: Bootstrap Confidence Intervals
=============================================
95% CIs for mean RMSE and improvement % for all methods,
bootstrapped over 229 timesteps (n=10,000 resamples).
Covers: Background, DINEOF, Cell-NN+Diffusion, Monte-Carlo (internal),
and Cell-NN+Diffusion vs VIIRS (external).
"""
import numpy as np

np.random.seed(42)
N_BOOT = 10000

print("=" * 65)
print("Experiment 2: Bootstrap Confidence Intervals")
print(f"n_bootstrap={N_BOOT}, resampling over 229 timesteps")
print("=" * 65)

def bootstrap_ci(rmse_arr, n_boot=N_BOOT, alpha=0.05):
    """Bootstrap 95% CI for mean RMSE from per-timestep array."""
    means = np.array([
        np.mean(np.random.choice(rmse_arr, size=len(rmse_arr), replace=True))
        for _ in range(n_boot)
    ])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return np.mean(rmse_arr), lo, hi

def bootstrap_improvement_ci(rmse_bg, rmse_method, n_boot=N_BOOT, alpha=0.05):
    """Bootstrap 95% CI for % improvement (bg - method) / bg."""
    n = len(rmse_bg)
    improvements = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        bg_m  = np.mean(rmse_bg[idx])
        met_m = np.mean(rmse_method[idx])
        improvements.append((bg_m - met_m) / bg_m * 100)
    lo = np.percentile(improvements, 100 * alpha / 2)
    hi = np.percentile(improvements, 100 * (1 - alpha / 2))
    mean_improv = (np.mean(rmse_bg) - np.mean(rmse_method)) / np.mean(rmse_bg) * 100
    return mean_improv, lo, hi

# ── Load per-timestep arrays ───────────────────────────────────────────
rmse_bg_int   = np.load('results/chl_proper_rmse_bg.npy')
rmse_cnn_int  = np.load('results/chl_proper_rmse_cellnn.npy')
rmse_bg_ext   = np.load('results/viirs_rmse_bg_final.npy')
rmse_cnn_ext  = np.load('results/viirs_rmse_cnn_final.npy')
rmse_mc       = np.load('results/montecarlo_causal_rmse.npy')
dineof_data   = np.load('results/dineof_results.npz')
rmse_dineof   = dineof_data['rmse_dineof']

# Align lengths (all should be 229 -- sanity check)
print(f"\nArray lengths (all should be 229):")
print(f"  BG internal:   {len(rmse_bg_int)}")
print(f"  CNN internal:  {len(rmse_cnn_int)}")
print(f"  BG external:   {len(rmse_bg_ext)}")
print(f"  CNN external:  {len(rmse_cnn_ext)}")
print(f"  MC causal:     {len(rmse_mc)}")
print(f"  DINEOF:        {len(rmse_dineof)}")

# Trim all to minimum length for safety
n_min = min(len(rmse_bg_int), len(rmse_cnn_int), len(rmse_mc),
            len(rmse_dineof), len(rmse_bg_ext), len(rmse_cnn_ext))
rmse_bg_int  = rmse_bg_int[:n_min]
rmse_cnn_int = rmse_cnn_int[:n_min]
rmse_mc      = rmse_mc[:n_min]
rmse_dineof  = rmse_dineof[:n_min]
rmse_bg_ext  = rmse_bg_ext[:n_min]
rmse_cnn_ext = rmse_cnn_ext[:n_min]

print(f"\nBootstrapping over {n_min} timesteps, {N_BOOT} resamples each...")

# ── INTERNAL VALIDATION ────────────────────────────────────────────────
print(f"\n--- INTERNAL VALIDATION (held-out MODIS pixels) ---")
methods_int = [
    ("Background",          rmse_bg_int),
    ("DINEOF (r=17)",       rmse_dineof),
    ("Cell-NN+Diffusion",   rmse_cnn_int),
    ("Monte-Carlo (causal)",rmse_mc),
]

ci_results_int = {}
print(f"{'Method':<26}{'Mean RMSE':>12}{'95% CI':>22}{'Improv%':>10}{'CI':>22}")
print("-"*95)
for name, arr in methods_int:
    mean, lo, hi = bootstrap_ci(arr)
    ci_results_int[name] = (mean, lo, hi)
    if name == "Background":
        print(f"{name:<26}{mean:>12.4f}  [{lo:.4f}, {hi:.4f}]{'---':>10}")
    else:
        improv, i_lo, i_hi = bootstrap_improvement_ci(rmse_bg_int, arr)
        print(f"{name:<26}{mean:>12.4f}  [{lo:.4f}, {hi:.4f}]"
              f"{improv:>9.1f}%  [{i_lo:.1f}%, {i_hi:.1f}%]")

# ── EXTERNAL VALIDATION ────────────────────────────────────────────────
print(f"\n--- EXTERNAL VALIDATION (independent VIIRS sensor) ---")
methods_ext = [
    ("Background",          rmse_bg_ext),
    ("Cell-NN+Diffusion",   rmse_cnn_ext),
]

ci_results_ext = {}
print(f"{'Method':<26}{'Mean RMSE':>12}{'95% CI':>22}{'Improv%':>10}{'CI':>22}")
print("-"*95)
for name, arr in methods_ext:
    mean, lo, hi = bootstrap_ci(arr)
    ci_results_ext[name] = (mean, lo, hi)
    if name == "Background":
        print(f"{name:<26}{mean:>12.4f}  [{lo:.4f}, {hi:.4f}]{'---':>10}")
    else:
        improv, i_lo, i_hi = bootstrap_improvement_ci(rmse_bg_ext, arr)
        print(f"{name:<26}{mean:>12.4f}  [{lo:.4f}, {hi:.4f}]"
              f"{improv:>9.1f}%  [{i_lo:.1f}%, {i_hi:.1f}%]")

# ── PAIRWISE SIGNIFICANCE ──────────────────────────────────────────────
print(f"\n--- PAIRWISE SIGNIFICANCE (is A better than B?) ---")
pairs = [
    ("DINEOF vs Background",        rmse_dineof,  rmse_bg_int),
    ("CNN+Diff vs Background",      rmse_cnn_int, rmse_bg_int),
    ("CNN+Diff vs DINEOF",          rmse_cnn_int, rmse_dineof),
    ("MC vs Background",            rmse_mc,      rmse_bg_int),
    ("MC vs CNN+Diff",              rmse_mc,      rmse_cnn_int),
]

for label, arr_a, arr_b in pairs:
    diffs = []
    n = len(arr_a)
    for _ in range(N_BOOT):
        idx = np.random.choice(n, size=n, replace=True)
        diffs.append(np.mean(arr_b[idx]) - np.mean(arr_a[idx]))
    diffs = np.array(diffs)
    p_better = np.mean(diffs > 0)   # P(A < B) = P(B - A > 0)
    lo = np.percentile(diffs, 2.5)
    hi = np.percentile(diffs, 97.5)
    sig = "***" if p_better > 0.999 else "**" if p_better > 0.99 else "*" if p_better > 0.95 else "ns"
    print(f"  {label:<35}  P(A<B)={p_better:.4f}  "
          f"diff CI=[{lo:.4f},{hi:.4f}]  {sig}")

print(f"\n*** p<0.001  ** p<0.01  * p<0.05  ns=not significant")

# ── Save ───────────────────────────────────────────────────────────────
np.savez('results/bootstrap_ci.npz',
         methods_internal=[m[0] for m in methods_int],
         ci_internal=np.array([ci_results_int[m[0]] for m in methods_int]),
         methods_external=[m[0] for m in methods_ext],
         ci_external=np.array([ci_results_ext[m[0]] for m in methods_ext]))
print("\nSaved -> results/bootstrap_ci.npz")
print("\nDone.")
