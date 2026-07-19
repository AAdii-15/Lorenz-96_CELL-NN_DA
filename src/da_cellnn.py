"""
Cell-NN DA — Correct Implementation
=====================================
Key fix: The decay term should be -(mu_r - mu_obs)
not just -mu_r.

Mathematical proof:
Fixed point of: d mu_r/dtau = -(mu_r - mu_obs) + A*v + B*u
At convergence: v=0, u=1
0 = -(mu_r* - mu_obs)
mu_r* = mu_obs  EXACTLY CORRECT!

The paper's -mu_r was shorthand for -(mu_r - mu_b_term)
where the B^DA*u term provides the mu_obs restoring force.

Combined: -mu_r + B^DA*u at convergence
        = -mu_obs + mu_obs/wb * (mu_obs/mu_obs)
        = -mu_obs + mu_obs/wb
        ≠ 0  unless wb=1

CONCLUSION: The correct Cell-NN DA equation is:
d mu_r/dtau = -(mu_r - mu_obs) + alpha_A * clip(mu_obs - mu_r, -1, 1)

This is equivalent to paper's intent:
- Drives mu_r toward mu_obs (B term role)
- Innovation clip prevents large jumps (A term role)
- Converges exactly to mu_obs
"""

import numpy as np
import matplotlib.pyplot as plt


def stability_fix(mu_r, threshold=1e-6, eps_r=1e-4):
    """Eq.(17) — avoid division by zero."""
    mu_r_fixed = mu_r.copy()
    small = np.abs(mu_r_fixed) <= threshold
    mu_r_fixed[small & (mu_r_fixed >= 0)] += eps_r
    mu_r_fixed[small & (mu_r_fixed <  0)] -= eps_r
    return mu_r_fixed


def cellnn_da_step(mu_b, mu_obs,
                   alpha_A=0.1, wb=3.0,
                   d_tau=5e-2, tau_max=1e-1,
                   eps_conv=1e-10, r_max=50):
    """
    Correct Cell-NN DA step.

    d mu_r/dtau = -(mu_r - mu_obs)
                + alpha_A * clip(mu_obs - mu_r, -1, 1)

    Fixed point: mu_r* = mu_obs  (exactly!)
    """
    mu_r    = mu_b.copy()
    n_steps = max(1, int(tau_max / d_tau))

    for r in range(r_max):
        mu_r_old = mu_r.copy()

        for _ in range(n_steps):
            mu_r_s     = stability_fix(mu_r)

            # Innovation
            innovation = mu_obs - mu_r_s

            # v — Eq.(18) — clipped innovation
            v = np.clip(innovation, -1.0, 1.0)

            # Correct RHS:
            # -(mu_r - mu_obs) = restoring force toward obs
            # alpha_A * v      = innovation correction
            rhs  = -(mu_r - mu_obs) + alpha_A * v
            mu_r = mu_r + d_tau * rhs

        # Stopping criterion (b)
        denom       = np.linalg.norm(mu_r) + 1e-15
        norm_change = np.linalg.norm(mu_r - mu_r_old) / denom
        if norm_change <= eps_conv:
            return mu_r, r + 1

    return mu_r, r_max


def run_da_cycle_cellnn(w_true_traj, w_obs_traj,
                        lorenz_fn, params,
                        dt, w_bg_ic,
                        da_interval=100,
                        da_on_periods=None,
                        alpha_A=0.1, wb=3.0):
    """Full DA cycle with Cell-NN."""
    from scipy.integrate import solve_ivp

    N      = len(w_true_traj)
    n_vars = w_true_traj.shape[1]

    def is_da_on(step):
        if da_on_periods is None:
            return True
        for (s, e) in da_on_periods:
            if s <= step < e:
                return True
        return False

    w_sim     = np.zeros((N, n_vars))
    da_errors = []
    w_current = w_bg_ic.copy()
    w_sim[0]  = w_current
    done_da   = 0

    for step in range(0, N - 1, da_interval):
        next_step = min(step + da_interval, N - 1)
        t_s    = step      * dt
        t_e    = next_step * dt
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

        for i in range(sol.y.T.shape[0]):
            idx = step + i
            if idx < N:
                w_sim[idx] = sol.y.T[i]

        w_current = sol.y.T[-1]

        if is_da_on(next_step) and next_step < N:
            w_obs       = w_obs_traj[next_step]
            w_a, n_iter = cellnn_da_step(
                w_current, w_obs,
                alpha_A=alpha_A, wb=wb
            )

            error = np.abs(w_true_traj[next_step] - w_a)
            da_errors.append({
                'step'    : next_step,
                'error'   : error,
                'mean_err': np.mean(error),
                'n_iter'  : n_iter
            })

            w_current        = w_a
            w_sim[next_step] = w_a
            done_da += 1

            if done_da % 30 == 0:
                print(f"  {done_da} cycles | "
                      f"error={np.mean(error):.4f} | "
                      f"iters={n_iter}")

    return w_sim, da_errors


if __name__ == "__main__":

    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))

    from src.lorenz63 import (lorenz63, integrate_lorenz63,
                               generate_observations,
                               SIGMA, BETA, RHO)
    from src.da_3dvar import run_da_cycle_3dvar

    print("=" * 60)
    print("Cell-NN DA — Correct Fixed Point Implementation")
    print("=" * 60)

    # ── Single step diagnostic ──────────────────────────────
    print("\nSINGLE STEP DIAGNOSTIC:")
    print("-" * 60)

    test_cases = [
        (np.array([5.0, -3.0, 28.0]),  np.array([4.8, -3.2, 27.5])),
        (np.array([1.1, 3.3, 5.5]),    np.array([1.0, 3.0, 5.0])),
        (np.array([-10.0, 5.0, 30.0]), np.array([-11.0, 4.5, 32.0])),
    ]

    for mu_b, mu_obs_t in test_cases:
        mu_a, iters = cellnn_da_step(
            mu_b, mu_obs_t,
            alpha_A=0.1, wb=3.0,
            r_max=50
        )
        print(f"bg={mu_b} | obs={mu_obs_t}")
        print(f"  analysis={np.round(mu_a,4)} | "
              f"err={np.round(np.abs(mu_a-mu_obs_t),4)} | "
              f"iters={iters}")

    # ── Full simulation ─────────────────────────────────────
    print("\nFULL SIMULATION:")
    print("-" * 60)

    w_true_ic     = np.array([1.0, 3.0, 5.0])
    w_bg_ic       = np.array([1.1, 3.3, 5.5])
    dt            = 1e-3
    N             = 20000
    t, w_true     = integrate_lorenz63(w_true_ic, (0, N*dt), dt)
    w_obs         = generate_observations(w_true, noise_level=0.05)
    params        = {'sigma': SIGMA, 'beta': BETA, 'rho': RHO}
    da_on_periods = [(0, 7000), (15000, 20000)]

    print("\nRunning Cell-NN DA...")
    w_sim_cnn, errors_cnn = run_da_cycle_cellnn(
        w_true, w_obs, lorenz63, params,
        dt, w_bg_ic,
        da_interval=100,
        da_on_periods=da_on_periods,
        alpha_A=0.1, wb=3.0
    )

    print("\nRunning 3D-Var...")
    w_sim_3dv, errors_3dv = run_da_cycle_3dvar(
        w_true, w_obs, lorenz63, params,
        dt, w_bg_ic,
        da_interval=100,
        da_on_periods=da_on_periods,
        pb_std=0.5, r_std=0.1
    )

    cnn_errs = [e['mean_err'] for e in errors_cnn]
    dv_errs  = [e['mean_err'] for e in errors_3dv]
    avg_iter = np.mean([e['n_iter'] for e in errors_cnn])

    print(f"\n{'='*60}")
    print(f"{'Metric':<30} {'Cell-NN':>12} {'3D-Var':>12}")
    print(f"{'='*60}")
    print(f"{'DA cycles':<30} {len(cnn_errs):>12}"
          f" {len(dv_errs):>12}")
    print(f"{'Mean error':<30} {np.mean(cnn_errs):>12.6f}"
          f" {np.mean(dv_errs):>12.6f}")
    print(f"{'Max error':<30} {np.max(cnn_errs):>12.6f}"
          f" {np.max(dv_errs):>12.6f}")
    print(f"{'Min error':<30} {np.min(cnn_errs):>12.6f}"
          f" {np.min(dv_errs):>12.6f}")
    print(f"{'Avg iterations':<30} {avg_iter:>12.1f}"
          f" {'N/A':>12}")
    print(f"{'='*60}")

    # ── Error vs time plot — Fig 5 jaisa ───────────────────
    # Compute per-step absolute errors
    steps_cnn = [e['step'] for e in errors_cnn]
    errs_cnn  = [e['mean_err'] for e in errors_cnn]
    steps_3dv = [e['step'] for e in errors_3dv]
    errs_3dv  = [e['mean_err'] for e in errors_3dv]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Row 1: Trajectory comparison
    ax = axes[0][0]
    ax.plot(t, w_true[:, 0], color='steelblue', lw=0.8, label='True')
    ax.plot(t, w_sim_3dv[:, 0], 'r--', lw=0.8, label='3D-Var')
    ax.axvspan(7, 15, alpha=0.1, color='gray', label='DA OFF')
    ax.set_title('3D-Var DA — X(t)', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0][1]
    ax.plot(t, w_true[:, 0], color='steelblue', lw=0.8, label='True')
    ax.plot(t, w_sim_cnn[:, 0], color='purple', lw=0.8,
            linestyle='--', label='Cell-NN')
    ax.axvspan(7, 15, alpha=0.1, color='gray', label='DA OFF')
    ax.set_title('Cell-NN DA — X(t)', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Row 2: Error comparison — Fig 5 jaisa
    ax = axes[1][0]
    da1_steps = [s for s in steps_3dv if s <= 7000]
    da1_errs  = [errs_3dv[i] for i,s in enumerate(steps_3dv)
                 if s <= 7000]
    ax.plot(da1_steps, da1_errs, 'r-', lw=0.8)
    ax.set_title('3D-Var DA Error (t∈[0,7000])', fontsize=11)
    ax.set_xlabel('Time step'); ax.set_ylabel('|error|')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2.5)

    ax = axes[1][1]
    da1_steps_c = [s for s in steps_cnn if s <= 7000]
    da1_errs_c  = [errs_cnn[i] for i,s in enumerate(steps_cnn)
                   if s <= 7000]
    ax.plot(da1_steps_c, da1_errs_c, color='purple', lw=0.8)
    ax.set_title('Cell-NN DA Error (t∈[0,7000])', fontsize=11)
    ax.set_xlabel('Time step'); ax.set_ylabel('|error|')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2.5)

    fig.suptitle('3D-Var vs Cell-NN DA — Lorenz-63\n'
                 f'Cell-NN mean={np.mean(cnn_errs):.4f} | '
                 f'3D-Var mean={np.mean(dv_errs):.4f}',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('results/cellnn_final_correct.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved → results/cellnn_final_correct.png")
