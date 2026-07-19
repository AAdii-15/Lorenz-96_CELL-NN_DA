"""
Cell-NN for Lorenz-96
======================
Template derivation:

Lorenz-96: dX_k/dt = (X_{k+1} - X_{k-2})*X_{k-1} - X_k + F

Cell-NN equation (A=B=0, I=F):
d mu_k/dt = -mu_k + sum_l C_{kl}*mu_l + sum_l D_{kl}*mu_l + F

Term by term comparison:
-X_k     → already in -mu_k  ✓
+F       → bias I = F         ✓

Nonlinear term:
(X_{k+1} - X_{k-2})*X_{k-1}
= X_{k+1}*X_{k-1} - X_{k-2}*X_{k-1}

This is D_{k,k-1} * mu_{k-1} where:
D_{k,k-1} = (X_{k+1} - X_{k-2})

All other C and D terms = 0

So:
C = 0  (no linear coupling)
D_{k,k-1} = (X_{k+1} - X_{k-2})  [nonlinear, time-varying]
I = F = 8
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def make_D_matrix(X, K=40):
    """
    Build D matrix for Lorenz-96 Cell-NN.

    D_{k, k-1} = X_{k+1} - X_{k-2}
    All other entries = 0.

    Parameters
    ----------
    X : array (K,) — current state
    K : int        — number of variables

    Returns
    -------
    D : array (K, K) — nonlinear template
    """
    D = np.zeros((K, K))

    for k in range(K):
        kp1  = (k + 1) % K   # k+1
        km2  = (k - 2) % K   # k-2
        km1  = (k - 1) % K   # k-1

        D[k, km1] = X[kp1] - X[km2]

    return D


def lorenz96_cellnn_rhs(t, mu, K=40, F=8.0):
    """
    Cell-NN RHS for Lorenz-96.

    d mu_k/dt = -mu_k + D(mu)_{k,k-1} * mu_{k-1} + F

    Vectorized implementation.

    Parameters
    ----------
    t   : float      — time
    mu  : array (K,) — current state
    K   : int        — number of variables
    F   : float      — forcing (bias I=F)

    Returns
    -------
    dmu : array (K,) — time derivative
    """
    # Build D matrix from current state
    D = make_D_matrix(mu, K)

    # Cell-NN equation:
    # d mu/dt = -mu + D*mu + F
    dmu = -mu + D @ mu + F

    return dmu


def lorenz96_cellnn_rhs_fast(t, mu, K=40, F=8.0):
    """
    Fast vectorized version — no matrix build.

    D_{k,k-1} * mu_{k-1} = (X_{k+1} - X_{k-2}) * X_{k-1}
    This is exactly the Lorenz-96 nonlinear term!
    """
    mup1 = np.roll(mu, -1)   # mu_{k+1}
    mum1 = np.roll(mu,  1)   # mu_{k-1}
    mum2 = np.roll(mu,  2)   # mu_{k-2}

    # d mu/dt = -mu + (mu_{k+1} - mu_{k-2})*mu_{k-1} + F
    return -mu + (mup1 - mum2) * mum1 + F


def integrate_lorenz96_cellnn(mu0, t_span, dt=1e-2,
                               K=40, F=8.0):
    """
    Integrate Lorenz-96 using Cell-NN formulation.
    Uses RK45 solver.

    Parameters
    ----------
    mu0    : array (K,)     — initial condition
    t_span : tuple (t0,tf)  — time interval
    dt     : float          — time step
    K      : int            — variables
    F      : float          — forcing

    Returns
    -------
    t   : array (N,)    — time points
    mu  : array (N, K)  — trajectory
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)

    sol = solve_ivp(
        fun=lambda t, mu: lorenz96_cellnn_rhs_fast(t, mu, K, F),
        t_span=t_span,
        y0=mu0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8
    )

    return sol.t, sol.y.T


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":

    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    from src.lorenz96 import (integrate_lorenz96,
                               make_initial_condition,
                               K_DEFAULT, F_DEFAULT)

    print("=" * 55)
    print("Cell-NN Time Integration — Lorenz-96")
    print("=" * 55)

    # Same initial condition
    X0    = make_initial_condition(K=40, F=8.0)
    t_span= (0, 5)
    dt    = 1e-2

    print(f"Integrating t∈[0,5], dt={dt}...")

    # Standard RK45
    t1, X_rk  = integrate_lorenz96(X0, t_span, dt)

    # Cell-NN
    t2, X_cnn = integrate_lorenz96_cellnn(X0, t_span, dt)

    print(f"RK45   shape: {X_rk.shape}")
    print(f"CellNN shape: {X_cnn.shape}")

    # Compare — should be IDENTICAL
    min_len = min(len(t1), len(t2))
    diff    = np.abs(X_rk[:min_len] - X_cnn[:min_len])

    print(f"\nComparison (should be ~0):")
    print(f"  Max difference : {np.max(diff):.2e}")
    print(f"  Mean difference: {np.mean(diff):.2e}")

    if np.max(diff) < 1e-4:
        print(f"  ✅ Cell-NN matches RK45 perfectly!")
    else:
        print(f"  ⚠️  Difference too large — check implementation")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Trajectory comparison — X_1
    axes[0][0].plot(t1[:300], X_rk[:300, 0],
                    'b-', lw=1.5, label='Standard RK45')
    axes[0][0].plot(t2[:300], X_cnn[:300, 0],
                    'r--', lw=1.0, label='Cell-NN')
    axes[0][0].set_title('X_1(t) — RK45 vs Cell-NN', fontsize=11)
    axes[0][0].legend(fontsize=9)
    axes[0][0].grid(True, alpha=0.3)

    # Trajectory comparison — X_20
    axes[0][1].plot(t1[:300], X_rk[:300, 19],
                    'b-', lw=1.5, label='Standard RK45')
    axes[0][1].plot(t2[:300], X_cnn[:300, 19],
                    'r--', lw=1.0, label='Cell-NN')
    axes[0][1].set_title('X_20(t) — RK45 vs Cell-NN', fontsize=11)
    axes[0][1].legend(fontsize=9)
    axes[0][1].grid(True, alpha=0.3)

    # Difference over time
    axes[1][0].semilogy(t1[:min_len],
                        np.mean(diff, axis=1), 'g-', lw=1)
    axes[1][0].set_title('Mean |RK45 - Cell-NN|', fontsize=11)
    axes[1][0].set_xlabel('Time')
    axes[1][0].grid(True, alpha=0.3)

    # Space-time of Cell-NN
    im = axes[1][1].contourf(
        np.arange(40), t2[:200],
        X_cnn[:200, :],
        levels=20, cmap='RdBu_r'
    )
    axes[1][1].set_title('Cell-NN Space-Time', fontsize=11)
    axes[1][1].set_xlabel('Variable k')
    axes[1][1].set_ylabel('Time')
    plt.colorbar(im, ax=axes[1][1])

    fig.suptitle('Lorenz-96: Cell-NN vs Standard Integration',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('results/lorenz96_cellnn_integration.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved → results/lorenz96_cellnn_integration.png")
