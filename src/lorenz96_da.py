"""
Lorenz-96 Data Assimilation
============================
Cell-NN DA + 3D-Var DA for Lorenz-96.
Same framework as Lorenz-63 -- just K=40 dimensions.

WHAT THIS SCRIPT DOES (high level):
  1. Simulates a "true" chaotic Lorenz-96 trajectory (the ground truth
     we're trying to track).
  2. Simulates a second, slightly-perturbed "background" trajectory
     (representing our imperfect starting guess).
  3. Every 25 timesteps, generates noisy partial observations of the
     TRUE trajectory (only 20 of 40 variables are ever observed).
  4. Runs TWO competing data assimilation (DA) methods that try to
     correct the background trajectory toward the true one using
     those noisy observations:
       - Cell-NN DA (the method proposed in this paper)
       - 3D-Var (a classical, well-established DA baseline)
  5. Compares how close each method's corrected trajectory stays to
     the true trajectory, both when DA is switched ON (observations
     arriving regularly) and OFF (no observations for a stretch of
     time, to test robustness).

This produces the numbers behind Table 2 and the figure behind
Figure 5 in the paper.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# lorenz96_fast        -- computes dX/dt for the Lorenz-96 equations
# integrate_lorenz96    -- runs a full trajectory forward in time (RK45)
# make_initial_condition -- creates a starting state vector, optionally
#                            "perturbed" (nudged away from a reference)
# generate_observations_96 -- takes a true trajectory and produces a
#                              noisy, partially-observed version of it
from src.lorenz96  import (lorenz96_fast, integrate_lorenz96,
                            make_initial_condition,
                            generate_observations_96,
                            K_DEFAULT, F_DEFAULT)

# cellnn_da_step -- the actual Cell-NN relaxation update (Eq. 1 in the
#                   paper): nudges a background value toward an
#                   observation using the clip-based relaxation rule.
from src.da_cellnn import cellnn_da_step

# analysis_3dvar -- classical 3D-Var analysis step (matrix-based,
#                   optimal linear combination of background + obs
#                   weighted by their respective error covariances).
# make_covariance_matrices -- builds the background-error (Pb) and
#                              observation-error (R) covariance
#                              matrices that 3D-Var needs.
from src.da_3dvar  import analysis_3dvar, make_covariance_matrices


# ══════════════════════════════════════════════════════════════
# 3D-Var DA for Lorenz-96
# ══════════════════════════════════════════════════════════════
# This is the CLASSICAL baseline method we compare Cell-NN against.
# 3D-Var computes the "best" correction by solving a weighted least-
# squares problem: it blends the background and the observation,
# weighting each by how much we trust it (via covariance matrices
# Pb and R). This requires knowing/estimating those covariances in
# advance -- Cell-NN, by contrast, needs no such covariance matrix.

def run_3dvar_lorenz96(X_true, X_obs, obs_idx,
                       X_bg_ic, dt,
                       da_interval=25,
                       da_on_periods=None,
                       pb_std=0.5, r_std=0.1,
                       K=40, F=8.0):
    """
    3D-Var DA cycle for Lorenz-96.
    Only observed variables get assimilated (the other 20/40
    variables are left to evolve purely from the model physics,
    with no direct correction -- they can still improve indirectly
    if the model's coupled dynamics propagate corrections to them).

    Parameters
    ----------
    X_true        : the true trajectory (ground truth), shape (N, K)
    X_obs         : noisy observations of X_true, same shape (N, K)
                     (only the columns in obs_idx are meaningful)
    obs_idx       : which of the K=40 variables we actually observe
    X_bg_ic       : the background trajectory's starting state
    dt            : integration timestep
    da_interval   : how many steps between each DA correction cycle
    da_on_periods : list of (start_step, end_step) tuples during which
                     DA corrections are actually applied. Outside these
                     windows, the model just free-runs with no
                     correction at all -- this is how we simulate
                     "observations stop arriving" (DA OFF).
    pb_std, r_std : standard deviations used to build the background-
                     error and observation-error covariance matrices
    """
    N = len(X_true)

    # Helper: is DA "switched on" at this particular step?
    # If da_on_periods is None, DA is always on (the simple case).
    # Otherwise, DA is only on during the listed (start, end) windows.
    def is_da_on(step):
        if da_on_periods is None:
            return True
        for (s, e) in da_on_periods:
            if s <= step < e:
                return True
        return False

    # Build the K x K background-error covariance (Pb) and the
    # observation-error covariance (R) -- these encode "how much do
    # we trust the background" vs "how much do we trust the
    # observations." 3D-Var needs these numbers explicitly; Cell-NN
    # does not use anything like this at all.
    Pb, R = make_covariance_matrices(K, pb_std, r_std)

    # Observation operator H: a matrix that picks out just the
    # observed variables from the full K-dimensional state vector.
    # H has one row per observed variable, with a single 1 in the
    # column corresponding to that variable's index.
    H = np.zeros((len(obs_idx), K))
    for i, k in enumerate(obs_idx):
        H[i, k] = 1.0

    # Observation-error covariance, restricted to just the observed
    # variables (a diagonal matrix -- we assume independent noise
    # across variables, each with standard deviation r_std).
    R_obs = np.eye(len(obs_idx)) * (r_std ** 2)

    w_sim     = np.zeros((N, K))   # will hold the full reconstructed
                                     # (analysis) trajectory over time
    da_errors = []                  # per-cycle error records
    w_current = X_bg_ic.copy()      # current state estimate, starts
                                     # at the (imperfect) background IC
    w_sim[0]  = w_current

    # Main loop: advance the state forward in chunks of da_interval
    # steps at a time. At the end of each chunk, optionally apply a
    # 3D-Var correction (if DA is "on" for that step).
    for step in range(0, N - 1, da_interval):
        next_step = min(step + da_interval, N - 1)
        t_s    = step      * dt
        t_e    = next_step * dt
        t_eval = np.linspace(t_s, t_e, next_step - step + 1)

        # --- FORECAST STEP ---
        # Integrate the Lorenz-96 equations forward from the current
        # state, using the real chaotic physics (RK45, high accuracy).
        # This happens EVERY cycle, regardless of whether DA is on or
        # off -- the model always keeps evolving forward in time.
        sol = solve_ivp(
            fun=lambda t, w: lorenz96_fast(t, w, K, F),
            t_span=(t_s, t_e),
            y0=w_current,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-8
        )

        # Save every intermediate timestep of this forecast segment
        # into the full trajectory array.
        for i in range(sol.y.T.shape[0]):
            idx = step + i
            if idx < N:
                w_sim[idx] = sol.y.T[i]

        w_current = sol.y.T[-1]  # state at the end of this forecast segment

        # --- ANALYSIS STEP (only if DA is switched on right now) ---
        if is_da_on(next_step) and next_step < N:
            # Extract just the observed variables' noisy values
            w_obs_full = X_obs[next_step]
            w_obs_part = w_obs_full[obs_idx]

            # Standard 3D-Var analysis formula:
            #   K_gain = (Pb^-1 + H^T R^-1 H)^-1 H^T R^-1
            #   w_a    = w_current + K_gain * (obs - H * w_current)
            # This is the classical "optimal" weighted blend of
            # background and observation, assuming both errors are
            # Gaussian with the specified covariances.
            Pb_inv  = np.linalg.inv(Pb)
            R_inv   = np.linalg.inv(R_obs)
            A       = Pb_inv + H.T @ R_inv @ H
            K_gain  = np.linalg.inv(A) @ H.T @ R_inv
            innov   = w_obs_part - H @ w_current   # "innovation": obs minus prediction
            w_a     = w_current + K_gain @ innov    # corrected (analysis) state

            # Record how far this correction landed from the TRUE state
            # (this is what Table 2's "Error" column is built from)
            error = np.abs(X_true[next_step] - w_a)
            da_errors.append({
                'step'    : next_step,
                'error'   : error,
                'mean_err': np.mean(error)
            })

            # The corrected state becomes the new "current" state,
            # and feeds into the next forecast segment.
            w_current        = w_a
            w_sim[next_step] = w_a

    return w_sim, da_errors


# ══════════════════════════════════════════════════════════════
# Cell-NN DA for Lorenz-96
# ══════════════════════════════════════════════════════════════
# This is the METHOD PROPOSED IN THIS PAPER. Unlike 3D-Var, it needs
# no covariance matrices at all -- just a single scalar parameter
# alpha_A. It works by "relaxing" each observed variable's background
# value toward its observation using the clip-based update rule
# (Eq. 1 in the paper), run to convergence.

def run_cellnn_lorenz96(X_true, X_obs, obs_idx,
                        X_bg_ic, dt,
                        da_interval=25,
                        da_on_periods=None,
                        alpha_A=1.0, wb=1.0,
                        K=40, F=8.0):
    """
    Cell-NN DA cycle for Lorenz-96.
    For unobserved variables -- use background (no correction).
    For observed variables   -- apply Cell-NN DA (the relaxation
    update from Eq. 1, run via cellnn_da_step()).

    Structurally this mirrors run_3dvar_lorenz96() above almost
    exactly (same forecast-then-analysis loop), except the analysis
    step uses the Cell-NN relaxation instead of the 3D-Var matrix
    formula. This makes the two methods directly, fairly comparable.
    """
    N = len(X_true)

    # Same DA on/off gating logic as the 3D-Var version above.
    def is_da_on(step):
        if da_on_periods is None:
            return True
        for (s, e) in da_on_periods:
            if s <= step < e:
                return True
        return False

    w_sim     = np.zeros((N, K))
    da_errors = []
    w_current = X_bg_ic.copy()
    w_sim[0]  = w_current
    done_da   = 0   # just a counter, used for progress printing

    for step in range(0, N - 1, da_interval):
        next_step = min(step + da_interval, N - 1)
        t_s    = step      * dt
        t_e    = next_step * dt
        t_eval = np.linspace(t_s, t_e, next_step - step + 1)

        # --- FORECAST STEP (identical logic to 3D-Var's version) ---
        # The chaotic Lorenz-96 physics always keeps running, whether
        # or not DA is currently switched on.
        sol = solve_ivp(
            fun=lambda t, w: lorenz96_fast(t, w, K, F),
            t_span=(t_s, t_e),
            y0=w_current,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-8
        )

        for i in range(sol.y.T.shape[0]):
            idx = step + i
            if idx < N:
                w_sim[idx] = sol.y.T[i]

        w_current = sol.y.T[-1]

        # --- ANALYSIS STEP: this is where Cell-NN differs from 3D-Var ---
        # THIS BLOCK ONLY RUNS IF is_da_on(next_step) IS TRUE.
        # If DA is "off" for this cycle, we skip straight to the next
        # loop iteration -- w_current simply keeps whatever value the
        # forecast step produced, completely uncorrected. This is the
        # exact mechanism behind "DA OFF": no special flag or reduced
        # alpha_A, just a total skip of the correction step.
        if is_da_on(next_step) and next_step < N:
            w_obs_full = X_obs[next_step]
            w_a        = w_current.copy()

            # Only the OBSERVED variables (obs_idx) get corrected.
            # The other 20 unobserved variables are left as whatever
            # the forecast produced -- any improvement they get comes
            # only indirectly, through the coupled Lorenz-96 dynamics
            # carrying the correction from observed neighbors into
            # them during the NEXT forecast segment.
            mu_b_obs   = w_current[obs_idx]   # background value at observed vars
            mu_obs_obs = w_obs_full[obs_idx]  # noisy observed value at those vars

            # THE CORE OF THE PAPER'S METHOD:
            # relax mu_b_obs toward mu_obs_obs using the clip-based
            # update rule (Eq. 1), iterated internally by
            # cellnn_da_step() until convergence (or until r_max
            # iterations are used up).
            mu_a_obs, n_iter = cellnn_da_step(
                mu_b_obs, mu_obs_obs,
                alpha_A=alpha_A, wb=wb
            )

            w_a[obs_idx] = mu_a_obs   # write the corrected values back
                                       # into the full 40-dim state vector

            # Record error against the TRUE state, same as 3D-Var does
            error = np.abs(X_true[next_step] - w_a)
            da_errors.append({
                'step'    : next_step,
                'error'   : error,
                'mean_err': np.mean(error),
                'n_iter'  : n_iter    # how many relaxation iterations
                                       # cellnn_da_step() actually used
            })

            w_current        = w_a
            w_sim[next_step] = w_a
            done_da += 1

            # Progress printout every 20 completed DA cycles
            if done_da % 20 == 0:
                print(f"  {done_da} cycles | "
                      f"error={np.mean(error):.4f} | "
                      f"iters={n_iter}")

    return w_sim, da_errors


# ══════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════
# This block actually runs everything: sets up the true/background
# trajectories, generates observations, runs both DA methods, prints
# the comparison table, and produces the two figures (main comparison
# + supplementary error space-time heatmaps).

if __name__ == "__main__":

    print("=" * 60)
    print("Lorenz-96 Data Assimilation")
    print("Cell-NN vs 3D-Var")
    print("=" * 60)

    # ── Setup ───────────────────────────────────────────────
    K   = 40    # number of state variables (Lorenz-96 dimension)
    F   = 8.0   # forcing constant in the Lorenz-96 equations
                #  (F=8.0 is the standard "strongly chaotic" choice)
    dt  = 1e-2  # integration timestep
    N   = 5000  # total number of timesteps = 50 time units

    # DA schedule: this is what creates the "DA ON / DA OFF" test.
    # Steps 0-2000   : DA ON  (observations arrive every da_interval steps)
    # Steps 2000-3500: DA OFF (no observations at all -- free chaotic drift)
    # Steps 3500-5000: DA ON  again (observations resume)
    # In real time units (dt=0.01): DA OFF spans t=20 to t=35.
    da_interval   = 25
    da_on_periods = [(0, 2000), (3500, 5000)]

    print(f"\nSetup:")
    print(f"  K={K}, F={F}, dt={dt}")
    print(f"  N={N} steps = {N*dt} time units")
    print(f"  DA interval : every {da_interval} steps")
    print(f"  DA ON  : [0,2000] and [3500,5000]")
    print(f"  DA OFF : (2000,3500)")

    # ── Initial conditions ───────────────────────────────────
    # X0_true: the TRUE starting state, no perturbation (seed=42).
    # X0_bg:   the BACKGROUND starting state -- a slightly different
    #          (5%-perturbed) state, using a different random seed
    #          (123). This represents our imperfect initial guess
    #          about where the system actually started.
    X0_true = make_initial_condition(K, F,
                                     perturb=0.0, seed=42)
    X0_bg   = make_initial_condition(K, F,
                                     perturb=0.05, seed=123)

    diff_ic = np.mean(np.abs(X0_true - X0_bg))
    print(f"\n  IC mean diff: {diff_ic:.4f} (~5% perturbation)")

    # ── True trajectory + observations ──────────────────────
    # Integrate the TRUE trajectory forward for the full N steps.
    # This is the "ground truth" both DA methods are trying to track.
    print("\nGenerating true trajectory...")
    t, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt)

    # Generate noisy, PARTIAL observations of the true trajectory:
    # only 20 of the 40 variables are ever observed (obs_every=2
    # means every other variable), and each observed value has 5%
    # noise added. This mimics a real sensor network that can't see
    # everything and whose readings aren't perfectly precise.
    print("Generating observations (20/40 variables)...")
    X_obs, obs_idx = generate_observations_96(
        X_true, noise_level=0.05, obs_every=2
    )
    print(f"  Observed: {len(obs_idx)}/40 variables")

    # ── Run Cell-NN DA ───────────────────────────────────────
    # Both DA methods start from the SAME background IC (X0_bg) and
    # see the SAME noisy observations (X_obs) -- this ensures a fair,
    # apples-to-apples comparison between them.
    print("\nRunning Cell-NN DA...")
    X_cnn, errors_cnn = run_cellnn_lorenz96(
        X_true, X_obs, obs_idx,
        X0_bg, dt,
        da_interval=da_interval,
        da_on_periods=da_on_periods,
        alpha_A=1.0, wb=1.0,   # optimized parameters from the grid
                                 # search in Figure 1/2 of the paper
        K=K, F=F
    )

    # ── Run 3D-Var DA ────────────────────────────────────────
    print("\nRunning 3D-Var DA...")
    X_3dv, errors_3dv = run_3dvar_lorenz96(
        X_true, X_obs, obs_idx,
        X0_bg, dt,
        da_interval=da_interval,
        da_on_periods=da_on_periods,
        pb_std=0.5, r_std=0.1,   # tuned covariance assumptions for 3D-Var
        K=K, F=F
    )

    # ── Results ─────────────────────────────────────────────
    # These are the numbers that populate Table 2 in the paper.
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

    # ── Main Figure 5: (a)-(d), 4 panels ───────────────────────
    # Row 1 (panels a,b): X_1(t) trajectory -- true vs each method's
    #   reconstruction. This shows visually how closely each method
    #   tracks the true chaotic trajectory of just ONE variable.
    # Row 2 (panels c,d): the corresponding error-over-time curves
    #   for each method.
    # The grey shaded band marks the DA-OFF window (t=20-35) in every
    # panel, so the reader can see exactly where the error grows
    # unconstrained.
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    panel_labels_row0 = ['(a)', '(b)']
    panel_labels_row1 = ['(c)', '(d)']

    # Row 1: X_1 trajectory -- (a) 3D-Var, (b) Cell-NN
    for col, (X_sim, name, lc) in enumerate([
        (X_3dv, '3D-Var', 'red'),
        (X_cnn, 'Cell-NN', 'purple')
    ]):
        axes[0][col].plot(t, X_true[:, 0],
                          'b-', lw=0.8, label='True')
        axes[0][col].plot(t, X_sim[:, 0],
                          color=lc, lw=0.8,
                          linestyle='--', label=name)
        # Grey shaded band = the DA-OFF window, in real time units
        axes[0][col].axvspan(20, 35, alpha=0.1,
                             color='gray', label='DA OFF')
        axes[0][col].set_title(f'{name} — X_1(t)', fontsize=11)
        axes[0][col].text(0.01, 0.95, panel_labels_row0[col],
                          transform=axes[0][col].transAxes,
                          fontsize=13, fontweight='bold', va='top')
        axes[0][col].legend(fontsize=8)
        axes[0][col].grid(True, alpha=0.3)

    # Row 2: Error over time -- (c) 3D-Var, (d) Cell-NN
    steps_cnn = [e['step'] for e in errors_cnn]
    steps_3dv = [e['step'] for e in errors_3dv]
    errs_cnn  = [e['mean_err'] for e in errors_cnn]
    errs_3dv  = [e['mean_err'] for e in errors_3dv]

    axes[1][0].plot([s*dt for s in steps_3dv],
                    errs_3dv, 'r-', lw=0.8)
    axes[1][0].set_title('3D-Var DA Error', fontsize=11)
    axes[1][0].text(0.01, 0.95, panel_labels_row1[0],
                    transform=axes[1][0].transAxes,
                    fontsize=13, fontweight='bold', va='top')
    axes[1][0].set_xlabel('Time')
    axes[1][0].set_ylabel('Mean |error|')
    axes[1][0].grid(True, alpha=0.3)

    axes[1][1].plot([s*dt for s in steps_cnn],
                    errs_cnn, color='purple', lw=0.8)
    axes[1][1].set_title('Cell-NN DA Error', fontsize=11)
    axes[1][1].text(0.01, 0.95, panel_labels_row1[1],
                    transform=axes[1][1].transAxes,
                    fontsize=13, fontweight='bold', va='top')
    axes[1][1].set_xlabel('Time')
    axes[1][1].set_ylabel('Mean |error|')
    axes[1][1].grid(True, alpha=0.3)

    fig.suptitle(
        f'Lorenz-96 DA: Cell-NN vs 3D-Var\n'
        f'Cell-NN mean={np.mean(cnn_errs):.4f} | '
        f'3D-Var mean={np.mean(dv_errs):.4f}',
        fontsize=12)
    plt.tight_layout()
    plt.savefig('results/lorenz96_da_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved → results/lorenz96_da_comparison.png")

    # ── Supplementary Figure 1: Error Space-Time heatmaps ──────
    # Instead of just showing ONE variable's error over time (like
    # Figure 5 does), this shows the error for ALL 40 variables at
    # once, as a 2D heatmap (variable index on the x-axis, time on
    # the y-axis). Only the first 200 timesteps are shown here to
    # keep the plot readable.
    fig_supp, axes_supp = plt.subplots(1, 2, figsize=(14, 5))
    for col, (X_sim, name) in enumerate([
        (X_3dv, '3D-Var'),
        (X_cnn, 'Cell-NN')
    ]):
        diff = np.abs(X_true - X_sim)
        im   = axes_supp[col].contourf(
            np.arange(K), t[:200],
            diff[:200, :],
            levels=20, cmap='hot_r'
        )
        axes_supp[col].set_title(
            f'{name} — |Error| Space-Time', fontsize=11)
        axes_supp[col].text(0.02, 0.95, ['(a)', '(b)'][col],
                             transform=axes_supp[col].transAxes,
                             fontsize=13, fontweight='bold', va='top',
                             color='black',
                             bbox=dict(boxstyle='round', facecolor='white',
                                       alpha=0.8, edgecolor='none'))
        axes_supp[col].set_xlabel('Variable k')
        axes_supp[col].set_ylabel('Time')
        plt.colorbar(im, ax=axes_supp[col])

    plt.tight_layout()
    plt.savefig('results/supplementary_fig1_error_spacetime.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Supplementary plot saved → "
          "results/supplementary_fig1_error_spacetime.png")

    # ── Save results ─────────────────────────────────────────
    # These .npy files are loaded by other scripts later (e.g. the
    # iteration-cap sweep, or anything reusing these error sequences)
    np.save('results/lorenz96_cnn_errors.npy',
            np.array(cnn_errs))
    np.save('results/lorenz96_3dv_errors.npy',
            np.array(dv_errs))
    print("Results saved → results/")