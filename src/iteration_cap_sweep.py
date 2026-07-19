"""
A2: Iteration-Cap (r_max) Sensitivity Sweep
=============================================
Sweeps r_max in {5, 10, 25, 50, 100, 200, 1000} on Lorenz-96, using
the EXACT primary-realization configuration from the paper:
  seed=(42,123), da_interval=25, alpha_A=1.0, wb=1.0, d_tau=5e-2

CRITICAL: tau_max is scaled proportionally with r_max to ensure
r_max remains the binding stopping constraint (not tau_max), since
cellnn_da_step's default tau_max=0.1 would otherwise terminate the
iteration after only ~2 steps for any r_max value, regardless of
the r_max parameter -- defeating the purpose of this sweep.

Reports:
  - R-score (DA-on periods only) for each r_max
  - Mean n_iter actually used (sanity check that r_max is binding)
  - Theoretical cumulative reduction factor from Theorem 1's Corollary:
    exp(-(1+alpha_A)*d_tau*r_max)
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src.lorenz96 import (integrate_lorenz96, make_initial_condition,
                           generate_observations_96)
from src.lorenz96_da import cellnn_da_step
from scipy.integrate import solve_ivp
from src.lorenz96 import lorenz96_fast
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("A2: Iteration-Cap (r_max) Sensitivity Sweep")
print("Primary realization: seed=(42,123), da_interval=25,")
print("alpha_A=1.0, wb=1.0, d_tau=5e-2")
print("=" * 70)

K, F, dt, N = 40, 8.0, 1e-2, 5000
DA_INTERVAL = 25
ALPHA_A, WB = 1.0, 1.0
D_TAU = 5e-2
R_MAX_VALUES = [5, 10, 25, 50, 100, 200, 1000]

da_on_periods = [(0, 2000), (3500, 5000)]
da_on_mask = np.zeros(N, dtype=bool)
for s, e in da_on_periods:
    da_on_mask[s:e] = True

def is_da_on(step):
    for (s, e) in da_on_periods:
        if s <= step < e:
            return True
    return False

def run_cellnn_lorenz96_rmax(X_true, X_obs, obs_idx, X_bg_ic, dt,
                              da_interval, da_on_periods,
                              alpha_A, wb, K, F, r_max, d_tau, tau_max):
    """Exact copy of run_cellnn_lorenz96, with r_max/d_tau/tau_max
    explicitly passed through to cellnn_da_step (not hardcoded)."""
    N = len(X_true)
    w_sim = np.zeros((N, K))
    da_errors = []
    w_current = X_bg_ic.copy()
    w_sim[0] = w_current

    for step in range(0, N - 1, da_interval):
        next_step = min(step + da_interval, N - 1)
        t_s, t_e = step * dt, next_step * dt
        t_eval = np.linspace(t_s, t_e, next_step - step + 1)

        sol = solve_ivp(
            fun=lambda t, w: lorenz96_fast(t, w, K, F),
            t_span=(t_s, t_e), y0=w_current, method='RK45',
            t_eval=t_eval, rtol=1e-8, atol=1e-8)

        for i in range(sol.y.T.shape[0]):
            idx = step + i
            if idx < N:
                w_sim[idx] = sol.y.T[i]
        w_current = sol.y.T[-1]

        if is_da_on(next_step) and next_step < N:
            w_obs_full = X_obs[next_step]
            w_a = w_current.copy()
            mu_b_obs = w_current[obs_idx]
            mu_obs_obs = w_obs_full[obs_idx]

            mu_a_obs, n_iter = cellnn_da_step(
                mu_b_obs, mu_obs_obs,
                alpha_A=alpha_A, wb=wb,
                d_tau=d_tau, tau_max=tau_max, r_max=r_max)

            w_a[obs_idx] = mu_a_obs
            error = np.abs(X_true[next_step] - w_a)
            da_errors.append({'step': next_step,
                              'mean_err': np.mean(error),
                              'n_iter': n_iter})
            w_current = w_a
            w_sim[next_step] = w_a

    return w_sim, da_errors

# ── Setup (matches paper's primary realization exactly) ────────────────
X0_true = make_initial_condition(K, F, perturb=0.0, seed=42)
X0_bg = make_initial_condition(K, F, perturb=0.05, seed=123)
_, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt)
np.random.seed(42)
X_obs, obs_idx = generate_observations_96(X_true, noise_level=0.05, obs_every=2)

print(f"\n{'r_max':>8}  {'R-score':>10}  {'mean n_iter':>12}  "
      f"{'r_max binding?':>15}  {'theory: frac. remaining':>24}")
print("-" * 80)

results = {}
for r_max in R_MAX_VALUES:
    # Scale tau_max generously so it never binds before r_max does
    tau_max = max(0.1, r_max * D_TAU * 2.0)

    w_sim, da_errors = run_cellnn_lorenz96_rmax(
        X_true, X_obs, obs_idx, X0_bg, dt,
        da_interval=DA_INTERVAL, da_on_periods=da_on_periods,
        alpha_A=ALPHA_A, wb=WB, K=K, F=F,
        r_max=r_max, d_tau=D_TAU, tau_max=tau_max)

    r_score = np.mean((X_true[da_on_mask] - w_sim[da_on_mask])**2)
    mean_n_iter = np.mean([e['n_iter'] for e in da_errors])
    binding = "YES" if abs(mean_n_iter - r_max) < 1 else \
              f"NO (used {mean_n_iter:.1f})"

    # Theoretical prediction from Theorem 1's Corollary
    theory_frac = np.exp(-(1 + ALPHA_A) * D_TAU * r_max)

    results[r_max] = {'r_score': r_score, 'mean_n_iter': mean_n_iter,
                      'theory_frac': theory_frac}

    print(f"{r_max:>8}  {r_score:>10.4f}  {mean_n_iter:>12.1f}  "
          f"{binding:>15}  {theory_frac:>24.6f}")

print(f"\n{'='*70}")
print("INTERPRETATION")
print(f"{'='*70}")
print("If r_max is the binding constraint at every tested value,")
print("'r_max binding?' should read YES throughout. If tau_max")
print("becomes binding for any r_max, this invalidates that point")
print("and tau_max scaling needs further adjustment.")
print()
print("Expected pattern: R-score should be worse (higher) at very")
print("small r_max (insufficient correction toward noisy observation)")
print("and may also degrade at very large r_max (full convergence to")
print("noisy observation each cycle, discarding background entirely).")
print("The current default r_max=50 should sit near the minimum.")

np.savez('results/iteration_cap_sweep.npz',
         r_max_values=np.array(R_MAX_VALUES),
         r_scores=np.array([results[r]['r_score'] for r in R_MAX_VALUES]),
         mean_n_iters=np.array([results[r]['mean_n_iter'] for r in R_MAX_VALUES]),
         theory_fracs=np.array([results[r]['theory_frac'] for r in R_MAX_VALUES]))
print("\nSaved -> results/iteration_cap_sweep.npz")
print("\nDone.")
