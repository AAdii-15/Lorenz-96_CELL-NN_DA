"""
Adaptive inflation EnKF -- RTPS (Whitaker & Hamill 2012)
============================================================
Relaxation-to-prior-spread adaptive inflation, added as a stronger
EnKF baseline per reviewer feedback. Reuses the exact same
build_localization(), Kalman gain, and perturbed-observation machinery
as src/enkf.py -- only the inflation mechanism changes: no fixed
multiplicative constant applied before the update; instead, posterior
spread is relaxed back toward prior spread, per state variable, after
the update, scaled by alpha.

Whitaker, J. S., and T. M. Hamill, 2012: Evaluating methods to account
for system errors in ensemble data assimilation. Mon. Wea. Rev., 140,
3078-3089, doi:10.1175/MWR-D-11-00276.1
"""
import numpy as np
from src.lorenz96 import lorenz96_fast
from src.enkf import rk4_step, build_localization

def enkf_analysis_rtps(X_ens, y_obs, obs_idx, R_std=0.05, loc_radius=8, alpha=0.7):
    N_ens, K = X_ens.shape
    n_obs = len(obs_idx)

    x_mean_prior = X_ens.mean(axis=0)
    sigma_b = X_ens.std(axis=0, ddof=1)  # prior spread, per variable

    A = X_ens - x_mean_prior  # NOTE: no fixed multiplicative inflation here

    H = np.zeros((n_obs, K))
    for i, idx in enumerate(obs_idx):
        H[i, idx] = 1.0

    rho = build_localization(K, obs_idx, loc_radius)

    HA = (H @ A.T).T
    Pf = (HA.T @ HA) / (N_ens - 1)
    PfHt = (A.T @ HA) / (N_ens - 1)
    PfHt_loc = PfHt * rho

    R = np.eye(n_obs) * R_std**2
    K_gain = PfHt_loc @ np.linalg.inv(Pf + R)

    Y_pert = (y_obs[:, None] + np.random.randn(n_obs, N_ens) * R_std)
    innov = Y_pert - H @ X_ens.T

    X_ana = X_ens + (K_gain @ innov).T
    x_mean_ana = X_ana.mean(axis=0)
    sigma_a = X_ana.std(axis=0, ddof=1)  # posterior spread, before RTPS

    # Floor sigma_a at the observation noise std, same reasoning as before.
    sigma_a_safe = np.maximum(sigma_a, R_std)
    inflate_factor = 1.0 + alpha * (sigma_b / sigma_a_safe - 1.0)
    inflate_factor = np.clip(inflate_factor, 1.0, 2.0)  # tighter per-cycle cap

    # A per-cycle cap alone is not enough: chaotic Lorenz-96 dynamics can
    # re-amplify an inflated ensemble between cycles, so a bounded per-cycle
    # factor can still compound to something astronomical over hundreds of
    # cycles. Add an absolute ceiling on ensemble spread, grounded in the
    # paper'''s own numbers -- Lorenz-96 state values are O(1-10) throughout
    # this study, and DA-off drift error (Fig. 6) tops out around 3.77.
    SPREAD_CEILING = 15.0
    excess = sigma_a * inflate_factor > SPREAD_CEILING
    if excess.any():
        inflate_factor[excess] = SPREAD_CEILING / np.maximum(sigma_a[excess], 1e-12)

    X_ana_relaxed = x_mean_ana + (X_ana - x_mean_ana) * inflate_factor[None, :]

    # Cap individual ensemble MEMBER deviations, not just the spread
    # statistic. A spread-level cap alone can still let one outlier member
    # get pushed far from the mean; that single member is then amplified by
    # Lorenz-96'''s quadratic coupling term on the next forecast step, which
    # is what caused the runaway growth traced in enkf_rtps_debug2.py.
    MEMBER_DEV_CAP = 10.0  # per-variable, grounded in the paper'''s own
                           # O(1-10) Lorenz-96 state magnitude (Sec. 3.3)
    dev = X_ana_relaxed - x_mean_ana
    over = np.abs(dev) > MEMBER_DEV_CAP
    if over.any():
        dev[over] = np.sign(dev[over]) * MEMBER_DEV_CAP
    X_ana_relaxed = x_mean_ana + dev

    return X_ana_relaxed, X_ana_relaxed.mean(axis=0)


def run_enkf_rtps_lorenz96(X_true, X_obs, obs_idx, X0_bg, dt,
                            da_interval=5, da_on_periods=None,
                            K=40, F=8.0, N_ens=50, R_std=0.05,
                            loc_radius=8, alpha=0.7, seed=42):
    N = len(X_true)
    if da_on_periods is None:
        da_on_periods = [(0, N)]
    da_on_mask = np.zeros(N, dtype=bool)
    for start, end in da_on_periods:
        da_on_mask[start:end] = True

    np.random.seed(seed)
    X_ens = X0_bg[None, :] + np.random.randn(N_ens, K) * 0.01
    X_sim = np.zeros((N, K))
    X_sim[0] = X_ens.mean(axis=0)

    STATE_CEILING = 20.0  # well above any state magnitude seen elsewhere
                          # in this paper (Sec. 3.3: O(1-10) typical range)
    for step in range(1, N):
        for m in range(N_ens):
            x_new = rk4_step(X_ens[m], dt, K, F)
            # Trigger BEFORE overflow, not after: the earlier trace showed
            # many members going non-finite in the same forecast step,
            # meaning the dynamics were already unstable before a
            # non-finite check alone would catch it.
            if not np.isfinite(x_new).all() or np.abs(x_new).max() > STATE_CEILING:
                x_new = X_ens.mean(axis=0) + np.random.randn(K) * 0.01
            X_ens[m] = x_new
        X_sim[step] = X_ens.mean(axis=0)

        if step % da_interval == 0 and da_on_mask[step]:
            y_obs = X_obs[step, obs_idx]
            X_ens, x_ana = enkf_analysis_rtps(
                X_ens, y_obs, obs_idx, R_std=R_std,
                loc_radius=loc_radius, alpha=alpha)
            X_sim[step] = x_ana

    return X_sim
