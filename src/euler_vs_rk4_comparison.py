"""
Euler vs RK4 ODE-score comparison
====================================
Reconstructs the "before/after" comparison referenced in the paper's
Diagnostics subsection: an earlier Euler-integration implementation of
the ODE-score reportedly overstated integration error by roughly six
orders of magnitude, compared to the corrected RK4-based scheme now
used throughout (compute_scores_maps.py, FIX 6).

This script uses the IDENTICAL RHS function (lorenz96_fast) and the
IDENTICAL true trajectory (X_true) as compute_scores_maps.py, so the
only difference between the two computed scores is the integrator
itself (forward Euler vs classical RK4) -- an apples-to-apples test.

ODE-score definition (one-step-ahead prediction MSE): starting from
each true state X_true[t], integrate forward one dt using the given
method, and compare the prediction to the true state X_true[t+1].
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath('.')))
from src.lorenz96 import integrate_lorenz96, make_initial_condition, lorenz96_fast
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("Euler vs RK4: ODE-score reconstruction")
print("=" * 65)

K, F, dt, N = 40, 8.0, 1e-2, 5000

# Identical setup to compute_scores_maps.py
X0_true = make_initial_condition(K, F, perturb=0.0, seed=42)
_, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt)

print(f"Trajectory shape: {X_true.shape}")

def compute_ode_score_euler(X_true, dt, F=8.0, K=40):
    """One-step-ahead prediction MSE using forward Euler."""
    errors = []
    for t in range(len(X_true) - 1):
        x_c = X_true[t]
        rhs = lorenz96_fast(0, x_c, K, F)
        x_pred = x_c + dt * rhs          # plain forward Euler
        err = np.mean((x_pred - X_true[t+1])**2)
        errors.append(err)
    return np.mean(errors)

def compute_ode_score_rk4(X_true, dt, F=8.0, K=40):
    """One-step-ahead prediction MSE using RK4 (identical to
    compute_scores_maps.py's compute_ode_score_rk4)."""
    errors = []
    for t in range(len(X_true) - 1):
        x_c = X_true[t]
        k1 = lorenz96_fast(0, x_c,              K, F)
        k2 = lorenz96_fast(0, x_c+0.5*dt*k1,   K, F)
        k3 = lorenz96_fast(0, x_c+0.5*dt*k2,   K, F)
        k4 = lorenz96_fast(0, x_c+dt*k3,        K, F)
        x_pred = x_c + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        err = np.mean((x_pred - X_true[t+1])**2)
        errors.append(err)
    return np.mean(errors)

print("\nComputing Euler-based ODE-score...")
ode_score_euler = compute_ode_score_euler(X_true, dt, F, K)
print(f"  Euler ODE-score: {ode_score_euler:.4e}")

print("\nComputing RK4-based ODE-score...")
ode_score_rk4 = compute_ode_score_rk4(X_true, dt, F, K)
print(f"  RK4 ODE-score:   {ode_score_rk4:.4e}")

ratio = ode_score_euler / ode_score_rk4
orders_of_magnitude = np.log10(ratio)

print(f"\n{'='*65}")
print("COMPARISON")
print(f"{'='*65}")
print(f"  Euler ODE-score : {ode_score_euler:.4e}")
print(f"  RK4 ODE-score   : {ode_score_rk4:.4e}")
print(f"  Ratio (Euler/RK4): {ratio:.4e}")
print(f"  Orders of magnitude difference: {orders_of_magnitude:.2f}")
print(f"{'='*65}")

print(f"\nPaper claim check:")
print(f"  Paper says Euler ~1e-7, RK4=1.79e-13, ~6 orders of magnitude")
print(f"  This run gives: Euler={ode_score_euler:.2e}, RK4={ode_score_rk4:.2e}, "
      f"{orders_of_magnitude:.2f} orders of magnitude")

np.savez('results/euler_vs_rk4_comparison.npz',
         ode_score_euler=ode_score_euler,
         ode_score_rk4=ode_score_rk4,
         ratio=ratio,
         orders_of_magnitude=orders_of_magnitude)
print("\nSaved -> results/euler_vs_rk4_comparison.npz")
print("\nDone.")
