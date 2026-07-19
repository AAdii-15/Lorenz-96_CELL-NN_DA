"""
Lorenz-96 System
================
dX_k/dt = (X_{k+1} - X_{k-2}) * X_{k-1} - X_k + F

Parameters:
    K = 40  (number of variables)
    F = 8   (forcing constant — chaotic regime)

Periodic boundary:
    X_{-1}  = X_{K-1}
    X_{0}   = X_{K}
    X_{K+1} = X_{1}
    X_{K+2} = X_{2}
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ── Standard parameters ───────────────────────────────────────
K_DEFAULT = 40
F_DEFAULT = 8.0


def lorenz96(t, X, K=K_DEFAULT, F=F_DEFAULT):
    """
    Lorenz-96 RHS — Eq. dX_k/dt = (X_{k+1} - X_{k-2})*X_{k-1} - X_k + F

    Parameters
    ----------
    t : float       — time (not used explicitly)
    X : array (K,)  — state vector
    K : int         — number of variables
    F : float       — forcing constant

    Returns
    -------
    dXdt : array (K,) — time derivatives
    """
    dXdt = np.zeros(K)

    for k in range(K):
        # Periodic boundary conditions
        kp1 = (k + 1) % K      # k+1
        km1 = (k - 1) % K      # k-1
        km2 = (k - 2) % K      # k-2

        dXdt[k] = (X[kp1] - X[km2]) * X[km1] - X[k] + F

    return dXdt


def lorenz96_fast(t, X, K=K_DEFAULT, F=F_DEFAULT):
    """
    Vectorized Lorenz-96 — same as above but faster.
    Uses numpy roll instead of loop.
    """
    Xp1 = np.roll(X, -1)   # X_{k+1}
    Xm1 = np.roll(X,  1)   # X_{k-1}
    Xm2 = np.roll(X,  2)   # X_{k-2}

    return (Xp1 - Xm2) * Xm1 - X + F


def integrate_lorenz96(X0, t_span, dt=1e-2,
                       K=K_DEFAULT, F=F_DEFAULT):
    """
    Integrate Lorenz-96 using RK45.

    Parameters
    ----------
    X0     : array (K,)     — initial condition
    t_span : tuple (t0, tf) — time interval
    dt     : float          — time step
    K      : int            — number of variables
    F      : float          — forcing

    Returns
    -------
    t      : array (N,)     — time points
    X      : array (N, K)   — state trajectory
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)

    sol = solve_ivp(
        fun=lambda t, X: lorenz96_fast(t, X, K, F),
        t_span=t_span,
        y0=X0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8
    )

    return sol.t, sol.y.T   # shape (N, K)


def make_initial_condition(K=K_DEFAULT, F=F_DEFAULT,
                           perturb=0.0, seed=42):
    """
    Standard Lorenz-96 initial condition.
    Start near equilibrium F, perturb one variable.

    Parameters
    ----------
    K       : int   — number of variables
    F       : float — forcing
    perturb : float — perturbation size (0 = standard)
    seed    : int   — random seed

    Returns
    -------
    X0 : array (K,) — initial condition
    """
    rng = np.random.default_rng(seed)
    X0  = np.ones(K) * F

    if perturb > 0:
        # Add random perturbation
        X0 += perturb * rng.uniform(-1, 1, K)
    else:
        # Standard: perturb only first variable
        X0[0] += 0.01

    return X0


def generate_observations_96(X_true, noise_level=0.05,
                              obs_every=2, seed=42):
    """
    Synthetic observations for Lorenz-96.
    Eq.(22) — same as paper.

    Parameters
    ----------
    X_true     : array (N, K) — true trajectory
    noise_level: float        — sigma_n^2 = 0.05
    obs_every  : int          — observe every nth variable
                                (2 = observe 20 out of 40)
    seed       : int          — random seed

    Returns
    -------
    X_obs      : array (N, K) — noisy observations
                                (NaN for unobserved variables)
    obs_idx    : array        — indices of observed variables
    """
    rng   = np.random.default_rng(seed)
    N, K  = X_true.shape

    # Observation indices
    obs_idx = np.arange(0, K, obs_every)   # 0,2,4,...,38

    # Initialize with NaN
    X_obs = np.full((N, K), np.nan)

    # Add noise only to observed variables
    nu    = rng.uniform(-1, 1, (N, len(obs_idx)))
    X_obs[:, obs_idx] = X_true[:, obs_idx] * (
        1 + noise_level * nu
    )

    return X_obs, obs_idx


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("Lorenz-96 Integration Test")
    print("=" * 55)
    print(f"K = {K_DEFAULT}, F = {F_DEFAULT}")
    print(f"Chaotic regime: F={F_DEFAULT} > 5.0 → YES")

    # Standard initial condition
    X0_true = make_initial_condition(K=40, F=8.0, perturb=0.0)
    X0_bg   = make_initial_condition(K=40, F=8.0, perturb=0.1)

    print(f"\nTrue IC  (first 5): {X0_true[:5]}")
    print(f"BG   IC  (first 5): {X0_bg[:5]}")

    # Integrate
    t_span = (0, 10)
    dt     = 1e-2
    print(f"\nIntegrating t∈[0,10], dt={dt}...")

    t, X_true = integrate_lorenz96(X0_true, t_span, dt)
    _, X_bg   = integrate_lorenz96(X0_bg,   t_span, dt)

    print(f"Done! Shape: {X_true.shape}")

    # Chaos check
    diff_init = np.mean(np.abs(X_true[0]  - X_bg[0]))
    diff_mid  = np.mean(np.abs(X_true[len(t)//2] - X_bg[len(t)//2]))
    diff_end  = np.mean(np.abs(X_true[-1] - X_bg[-1]))

    print(f"\nDivergence check:")
    print(f"  t=0   : {diff_init:.6f}")
    print(f"  t=5   : {diff_mid:.6f}")
    print(f"  t=10  : {diff_end:.6f}")
    print(f"  Chaos : {'YES' if diff_end > diff_init*10 else 'NO'}")

    # Observations
    X_obs, obs_idx = generate_observations_96(X_true,
                                               noise_level=0.05,
                                               obs_every=2)
    print(f"\nObservations:")
    print(f"  Total variables   : {K_DEFAULT}")
    print(f"  Observed variables: {len(obs_idx)}")
    print(f"  Observed indices  : {obs_idx[:5]}...{obs_idx[-5:]}")

    # Plot — space-time diagram
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Hovmoller diagram — True
    im1 = axes[0].contourf(
        np.arange(K_DEFAULT), t[:200],
        X_true[:200, :],
        levels=20, cmap='RdBu_r'
    )
    axes[0].set_xlabel('Variable k', fontsize=11)
    axes[0].set_ylabel('Time', fontsize=11)
    axes[0].set_title('Lorenz-96: True State\n'
                      '(Space-Time Diagram)', fontsize=11)
    plt.colorbar(im1, ax=axes[0])

    # Time series of first 5 variables
    for k in range(5):
        axes[1].plot(t[:500], X_true[:500, k],
                     lw=0.8, label=f'X_{k+1}')
    axes[1].set_xlabel('Time', fontsize=11)
    axes[1].set_ylabel('X_k', fontsize=11)
    axes[1].set_title('Lorenz-96: First 5 Variables', fontsize=11)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/lorenz96_test.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved → results/lorenz96_test.png")
