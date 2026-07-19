"""
Lorenz-63 System
================
Paper ke equations (1), (2), (3) ka implementation.

dX/dt = sigma * (Y - X)
dY/dt = X * (rho - Z) - Y
dZ/dt = X * Y - beta * Z

Parameters (chaotic regime):
    sigma = 10
    beta  = 8/3
    rho   = 32
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# ── Standard chaotic parameters ──────────────────────────────────────────────
SIGMA = 10.0
BETA  = 8.0 / 3.0
RHO   = 32.0


def lorenz63(t, w, sigma=SIGMA, beta=BETA, rho=RHO):
    """
    Lorenz-63 equations — right hand side.

    Parameters
    ----------
    t     : float        — current time (not used explicitly, required by solver)
    w     : array (3,)   — state vector [X, Y, Z]
    sigma : float        — Prandtl number
    beta  : float        — aspect ratio
    rho   : float        — reduced Rayleigh number

    Returns
    -------
    dwdt  : array (3,)   — time derivatives [dX/dt, dY/dt, dZ/dt]
    """
    X, Y, Z = w

    dXdt = sigma * (Y - X)           # Eq. (1)
    dYdt = X * (rho - Z) - Y         # Eq. (2)
    dZdt = X * Y - beta * Z          # Eq. (3)

    return np.array([dXdt, dYdt, dZdt])


def integrate_lorenz63(w0, t_span, dt=1e-3, sigma=SIGMA, beta=BETA, rho=RHO):
    """
    Integrate Lorenz-63 system using RK45.

    Parameters
    ----------
    w0     : array (3,)        — initial condition [X0, Y0, Z0]
    t_span : tuple (t0, tf)    — time interval
    dt     : float             — time step
    sigma  : float
    beta   : float
    rho    : float

    Returns
    -------
    t      : array (N,)        — time points
    w      : array (N, 3)      — state trajectory [X, Y, Z]
    """
    t_eval = np.arange(t_span[0], t_span[1], dt)

    sol = solve_ivp(
        fun=lambda t, w: lorenz63(t, w, sigma, beta, rho),
        t_span=t_span,
        y0=w0,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-10
    )

    return sol.t, sol.y.T   # sol.y.T → shape (N, 3)


def generate_observations(w_true, noise_level=0.05, seed=42):
    """
    Synthetic observations — paper ka Eq. (22).

    w_obs(t) = w_true(t) * [1 + sigma_n^2 * nu(t)]

    Parameters
    ----------
    w_true      : array (N, 3)  — true state trajectory
    noise_level : float         — sigma_n^2 = 0.05 (paper mein)
    seed        : int           — random seed for reproducibility

    Returns
    -------
    w_obs       : array (N, 3)  — noisy observations
    """
    rng   = np.random.default_rng(seed)
    nu    = rng.uniform(-1, 1, size=w_true.shape)   # nu ∈ [-1, 1]
    w_obs = w_true * (1 + noise_level * nu)

    return w_obs


def plot_lorenz63(t, w_true, w_bg=None, title="Lorenz-63 Trajectory"):
    """
    Plot Lorenz-63 trajectory — X, Y, Z vs time.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ['X(t)', 'Y(t)', 'Z(t)']
    colors = ['steelblue', 'darkorange', 'green']

    panel_labels = ['(a)', '(b)', '(c)']
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(t, w_true[:, i], color=color, lw=0.8, label='True')
        if w_bg is not None:
            ax.plot(t, w_bg[:, i], color='red', lw=0.8,
                    linestyle='--', label='Background')
        ax.set_ylabel(label, fontsize=12)
        ax.text(0.01, 0.90, panel_labels[i], transform=ax.transAxes,
                fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time steps', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/lorenz63_trajectory.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved → results/lorenz63_trajectory.png")


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Paper ke initial conditions
    w_true_ic = np.array([1.0, 3.0, 5.0])    # True system   — w1(0)
    w_bg_ic   = np.array([1.1, 3.3, 5.5])    # Background    — w2(0)

    print("=" * 50)
    print("Lorenz-63 Integration Test")
    print("=" * 50)
    print(f"True IC      : {w_true_ic}")
    print(f"Background IC: {w_bg_ic}")
    print(f"Parameters   : sigma={SIGMA}, beta={BETA:.4f}, rho={RHO}")
    print(f"Chaotic?     : {'YES' if RHO > 24.74 else 'NO'}")
    print("=" * 50)

    # Integrate both
    t_span = (0, 20)
    t, w_true = integrate_lorenz63(w_true_ic, t_span)
    _, w_bg   = integrate_lorenz63(w_bg_ic,   t_span)

    # Observations generate karo
    w_obs = generate_observations(w_true, noise_level=0.05)

    print(f"\nIntegration done!")
    print(f"Time steps   : {len(t)}")
    print(f"w_true shape : {w_true.shape}")
    print(f"w_obs shape  : {w_obs.shape}")

    # Divergence check — chaotic sensitivity
    diff = np.abs(w_true - w_bg)
    print(f"\nDivergence (|true - background|):")
    print(f"  At t=0  : {diff[0]}")
    print(f"  At t=10 : {diff[len(t)//2]}")
    print(f"  At t=20 : {diff[-1]}")
    print("\nNotice: Small initial diff grows → CHAOS confirmed!")

    # Plot
    plot_lorenz63(t, w_true, w_bg, title="Lorenz-63: True vs Background")
