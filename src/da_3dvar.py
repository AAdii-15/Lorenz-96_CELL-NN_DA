"""
3D-Var Data Assimilation
========================
Paper ka Section 3.1 aur Eq. (9), (23), (24), (25) ka implementation.

CORRECT DA CYCLE:
1. Background se integrate karo da_interval steps
2. Analysis compute karo
3. Analysis se wapas integrate karo next da_interval steps
4. Repeat!
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def make_covariance_matrices(n_vars=3, pb_std=0.5, r_std=0.1):
    """
    Background aur Observation error covariance matrices.
    Paper ke Eqs. (23) aur (24).
    """
    Pb = np.eye(n_vars) * (pb_std ** 2)
    R  = np.eye(n_vars) * (r_std  ** 2)
    return Pb, R


def analysis_3dvar(w_b, w_obs, Pb, R, H=None):
    """
    3D-Var analysis — paper ka Eq. (25).
    w_a = w_b + K * (w_obs - H * w_b)
    """
    n = len(w_b)
    if H is None:
        H = np.eye(n)

    Pb_inv     = np.linalg.inv(Pb)
    R_inv      = np.linalg.inv(R)
    A          = Pb_inv + H.T @ R_inv @ H
    K          = np.linalg.inv(A) @ H.T @ R_inv
    innovation = w_obs - H @ w_b
    w_a        = w_b + K @ innovation

    return w_a


def run_da_cycle_3dvar(w_true_traj, w_obs_traj,
                       lorenz_fn, params,
                       dt, w_bg_ic,
                       da_interval=100,
                       da_on_periods=None,
                       pb_std=0.5, r_std=0.1):
    """
    Correct DA cycle — analysis se wapas integrate karta hai.

    Parameters
    ----------
    w_true_traj  : array (N, n)     — true trajectory
    w_obs_traj   : array (N, n)     — observations
    lorenz_fn    : function         — RHS of dynamical system
    params       : dict             — model parameters
    dt           : float            — time step
    w_bg_ic      : array (n,)       — background initial condition
    da_interval  : int              — DA every N steps (paper: 100)
    da_on_periods: list of tuples   — [(start1,end1), (start2,end2)]
                                      None = always on
    pb_std       : float            — background error std
    r_std        : float            — observation error std

    Returns
    -------
    w_sim        : array (N, n)     — simulated trajectory with DA
    da_errors    : list             — error at each DA step
    """
    N      = len(w_true_traj)
    n_vars = w_true_traj.shape[1]
    Pb, R  = make_covariance_matrices(n_vars, pb_std, r_std)

    # DA on/off helper
    def is_da_on(step):
        if da_on_periods is None:
            return True
        for (s, e) in da_on_periods:
            if s <= step < e:
                return True
        return False

    # Storage
    w_sim     = np.zeros((N, n_vars))
    da_errors = []

    # Start from background IC
    w_current = w_bg_ic.copy()
    w_sim[0]  = w_current

    for step in range(0, N - 1, da_interval):

        next_step = min(step + da_interval, N - 1)

        # ── Integrate step → next_step ──────────────────────────
        t_s    = step      * dt
        t_e    = next_step * dt

        # t_eval must be strictly within t_span
        t_eval = np.linspace(t_s, t_e, next_step - step + 1)

        sol = solve_ivp(
            fun=lambda t, w: lorenz_fn(t, w, **params),
            t_span=(t_s, t_e),
            y0=w_current,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-10
        )

        # Store trajectory segment
        seg_len = sol.y.T.shape[0]
        for i in range(seg_len):
            idx = step + i
            if idx < N:
                w_sim[idx] = sol.y.T[i]

        # Update current state
        w_current = sol.y.T[-1]

        # ── DA at next_step ─────────────────────────────────────
        if is_da_on(next_step) and next_step < N:
            w_obs = w_obs_traj[next_step]
            w_a   = analysis_3dvar(w_current, w_obs, Pb, R)

            # Error vs true
            error = np.abs(w_true_traj[next_step] - w_a)
            da_errors.append({
                'step'    : next_step,
                'error'   : error,
                'mean_err': np.mean(error)
            })

            # Continue from analysis!
            w_current        = w_a
            w_sim[next_step] = w_a

    return w_sim, da_errors


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.lorenz63 import (lorenz63, integrate_lorenz63,
                               generate_observations,
                               SIGMA, BETA, RHO)

    print("=" * 50)
    print("3D-Var Data Assimilation Test")
    print("=" * 50)

    # Paper ke settings
    w_true_ic = np.array([1.0, 3.0, 5.0])
    w_bg_ic   = np.array([1.1, 3.3, 5.5])
    dt        = 1e-3
    N         = 20000

    # True trajectory + observations
    t, w_true = integrate_lorenz63(w_true_ic, (0, N * dt), dt)
    w_obs     = generate_observations(w_true, noise_level=0.05)
    params    = {'sigma': SIGMA, 'beta': BETA, 'rho': RHO}

    print(f"Total steps : {N}")
    print(f"DA interval : 100 steps")
    print(f"DA ON  : [0, 7000] and [15000, 20000]")
    print(f"DA OFF : (7000, 15000)")

    # Paper jaisa schedule — Eq. DA on/off
    da_on_periods = [(0, 7000), (15000, 20000)]

    print("\nRunning 3D-Var DA...")
    w_sim, da_errors = run_da_cycle_3dvar(
        w_true, w_obs,
        lorenz63, params,
        dt, w_bg_ic,
        da_interval=100,
        da_on_periods=da_on_periods,
        pb_std=0.5, r_std=0.1
    )

    # Results
    mean_errors = [e['mean_err'] for e in da_errors]
    print(f"\n3D-Var Results:")
    print(f"  Total DA cycles  : {len(da_errors)}")
    print(f"  Mean DA error    : {np.mean(mean_errors):.6f}")
    print(f"  Max  DA error    : {np.max(mean_errors):.6f}")
    print(f"  Min  DA error    : {np.min(mean_errors):.6f}")

    # Plot — paper ka Fig 4 jaisa
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ['X(t)', 'Y(t)', 'Z(t)']
    colors = ['steelblue', 'darkorange', 'green']

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(t, w_true[:, i],
                color=color, lw=0.8, label='True')
        ax.plot(t, w_sim[:, i],
                color='red', lw=0.8, linestyle='--', label='3D-Var')
        ax.axvspan(7*1, 15*1, alpha=0.1, color='gray', label='DA OFF')
        ax.set_ylabel(label, fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time', fontsize=11)
    fig.suptitle('3D-Var DA: True vs Analysis', fontsize=13)
    plt.tight_layout()
    plt.savefig('results/3dvar_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved → results/3dvar_analysis.png")
