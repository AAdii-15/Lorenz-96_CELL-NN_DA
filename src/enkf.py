"""
Ensemble Kalman Filter with Localization for Lorenz-96
Localization prevents filter divergence in chaotic systems
"""
import numpy as np
from src.lorenz96 import lorenz96_fast

def rk4_step(x, dt, K=40, F=8.0):
    k1 = lorenz96_fast(0, x, K, F)
    k2 = lorenz96_fast(0, x + 0.5*dt*k1, K, F)
    k3 = lorenz96_fast(0, x + 0.5*dt*k2, K, F)
    k4 = lorenz96_fast(0, x + dt*k3, K, F)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def gaspari_cohn(dist, radius):
    """
    Gaspari-Cohn localization function.
    Standard localization used in operational EnKF.
    Returns 1 at dist=0, 0 at dist>=2*radius.
    """
    r = np.abs(dist) / radius
    rho = np.zeros_like(r, dtype=float)
    # Inner part (r <= 1)
    m = r <= 1
    rho[m] = (-(r[m]**5)/4 + (r[m]**4)/2 +
               (r[m]**3)*5/8 - (r[m]**2)*5/3 + 1)
    # Outer part (1 < r <= 2)
    m = (r > 1) & (r <= 2)
    rho[m] = ((r[m]**5)/12 - (r[m]**4)/2 +
               (r[m]**3)*5/8 + (r[m]**2)*5/3 -
               5*r[m] + 4 - 2/(3*r[m]))
    return rho

def build_localization(K, obs_idx, radius=10):
    """
    Build localization matrix rho (K x n_obs).
    Reduces covariance for distant variables.
    """
    n_obs = len(obs_idx)
    rho   = np.zeros((K, n_obs))
    for j, oi in enumerate(obs_idx):
        for k in range(K):
            # Periodic distance on circle
            dist = min(abs(k - oi), K - abs(k - oi))
            rho[k, j] = gaspari_cohn(
                np.array([dist]), radius)[0]
    return rho

def enkf_analysis(X_ens, y_obs, obs_idx,
                  R_std=0.05, inflation=1.05,
                  loc_radius=8):
    N_ens, K = X_ens.shape
    n_obs    = len(obs_idx)

    # Covariance inflation
    x_mean = X_ens.mean(axis=0)
    A      = (X_ens - x_mean) * inflation
    X_ens  = x_mean + A

    # Observation operator
    H = np.zeros((n_obs, K))
    for i, idx in enumerate(obs_idx):
        H[i, idx] = 1.0

    # Localization matrix
    rho = build_localization(K, obs_idx, loc_radius)

    # Covariances
    HA   = (H @ A.T).T               # (N_ens, n_obs)
    Pf   = (HA.T @ HA) / (N_ens-1)  # (n_obs, n_obs)
    PfHt = (A.T @ HA) / (N_ens-1)   # (K, n_obs)

    # Apply localization to PfHt
    PfHt_loc = PfHt * rho            # (K, n_obs)

    R      = np.eye(n_obs) * R_std**2
    K_gain = PfHt_loc @ np.linalg.inv(Pf + R)

    # Perturbed obs
    Y_pert = (y_obs[:, None] +
              np.random.randn(n_obs, N_ens) * R_std)
    innov  = Y_pert - H @ X_ens.T

    X_ana  = X_ens + (K_gain @ innov).T
    return X_ana, X_ana.mean(axis=0)


def run_enkf_lorenz96(X_true, X_obs, obs_idx,
                      X0_bg, dt,
                      da_interval=5,
                      da_on_periods=None,
                      K=40, F=8.0,
                      N_ens=50,
                      R_std=0.05,
                      inflation=1.05,
                      loc_radius=8):
    N = len(X_true)
    if da_on_periods is None:
        da_on_periods = [(0, N)]

    da_on_mask = np.zeros(N, dtype=bool)
    for start, end in da_on_periods:
        da_on_mask[start:end] = True

    np.random.seed(42)
    X_ens    = X0_bg[None,:] + np.random.randn(N_ens, K) * 0.01
    X_sim    = np.zeros((N, K))
    X_sim[0] = X_ens.mean(axis=0)
    errors   = []

    print(f"  Running EnKF (N_ens={N_ens}, "
          f"infl={inflation}, loc_r={loc_radius}, "
          f"da_int={da_interval})...")

    for step in range(1, N):
        for m in range(N_ens):
            x_new = rk4_step(X_ens[m], dt, K, F)
            if not np.isfinite(x_new).all():
                x_new = (X_ens.mean(axis=0) +
                         np.random.randn(K) * 0.01)
            X_ens[m] = x_new

        X_sim[step] = X_ens.mean(axis=0)

        if step % da_interval == 0 and da_on_mask[step]:
            y_obs = X_obs[step, obs_idx]
            X_ens, x_ana = enkf_analysis(
                X_ens, y_obs, obs_idx,
                R_std=R_std,
                inflation=inflation,
                loc_radius=loc_radius)
            X_sim[step] = x_ana

            err = np.mean(np.abs(X_true[step] - x_ana))
            errors.append({'step': step, 'mean_err': err})

            if step % (da_interval * 100) == 0:
                print(f"    step={step:4d} | "
                      f"error={err:.4f} | "
                      f"spread={X_ens.std():.3f}")

    return X_sim, errors
