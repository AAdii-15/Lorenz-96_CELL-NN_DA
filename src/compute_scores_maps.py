"""
Fixed R-score, ODE-score (RK4) + Spatial Maps
FIX 6: ODE-score now uses RK4 instead of Euler
FIX 2: R-score only during DA ON periods
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lorenz96 import (integrate_lorenz96, make_initial_condition,
                           generate_observations_96, lorenz96_fast)
from src.lorenz96_da import run_cellnn_lorenz96, run_3dvar_lorenz96

print("="*60)
print("Computing R-score (DA ON only) + ODE-score (RK4)")
print("="*60)

K   = 40
F   = 8.0
dt  = 1e-2
N   = 5000

X0_true = make_initial_condition(K, F, perturb=0.0,  seed=42)
X0_bg   = make_initial_condition(K, F, perturb=0.05, seed=123)
t, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt)

# FIXED: noise_level=0.05 (not 0.025)
X_obs, obs_idx = generate_observations_96(X_true, noise_level=0.05, obs_every=2)
da_on = [(0, 2000), (3500, 5000)]

print("Running Cell-NN DA (αA=1.0, wb=1.0)...")
X_cnn, err_cnn = run_cellnn_lorenz96(
    X_true, X_obs, obs_idx, X0_bg, dt,
    da_interval=25, da_on_periods=da_on,
    alpha_A=1.0, wb=1.0, K=K, F=F
)

print("Running 3D-Var DA...")
X_3dv, err_3dv = run_3dvar_lorenz96(
    X_true, X_obs, obs_idx, X0_bg, dt,
    da_interval=25, da_on_periods=da_on,
    K=K, F=F
)

# FIX 2: R-score — DA ON periods ONLY
da_on_mask = np.zeros(N, dtype=bool)
for start, end in da_on:
    da_on_mask[start:end] = True

r_score_cnn = np.mean((X_true[da_on_mask] - X_cnn[da_on_mask])**2)
r_score_3dv = np.mean((X_true[da_on_mask] - X_3dv[da_on_mask])**2)

print(f"\nR-score (MSE during DA ON only):")
print(f"  Cell-NN : {r_score_cnn:.4f}")
print(f"  3D-Var  : {r_score_3dv:.4f}")

# FIX 6: ODE-score — RK4 (not Euler)
def compute_ode_score_rk4(X_true, dt, F=8.0, K=40):
    """
    One-step-ahead prediction MSE using RK4.
    Same as Fablet 2021 — RK4 integration.
    """
    errors = []
    for i in range(len(X_true)-1):
        x_c = X_true[i]
        k1  = lorenz96_fast(0, x_c,             K, F)
        k2  = lorenz96_fast(0, x_c+0.5*dt*k1,  K, F)
        k3  = lorenz96_fast(0, x_c+0.5*dt*k2,  K, F)
        k4  = lorenz96_fast(0, x_c+dt*k3,       K, F)
        x_p = x_c + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        errors.append(np.mean((X_true[i+1] - x_p)**2))
    return np.mean(errors)

print("\nComputing ODE-score (RK4)...")
ode_score = compute_ode_score_rk4(X_true, dt, F, K)
print(f"ODE-score (RK4): {ode_score:.2e}")

# ── Summary Table ─────────────────────────────────────
print(f"\n{'='*65}")
print(f"COMPARISON TABLE (like Fablet 2021 Table 2)")
print(f"{'='*65}")
print(f"{'Method':<32} {'R-score':>10} {'ODE-score':>12}")
print(f"{'-'*65}")
print(f"{'4DVar baseline (Fablet)':<32} {'1.06':>10} {'<1e-4':>12}")
print(f"{'Unsup CNN (Fablet)':<32} {'1.00':>10} {'<1e-4':>12}")
print(f"{'Sup GENN+LSTM (Fablet)':<32} {'0.38':>10} {'7.2e-2':>12}")
print(f"{'-'*65}")
print(f"{'3D-Var (ours)':<32} {r_score_3dv:>10.4f} {ode_score:>12.2e}")
print(f"{'Cell-NN (ours)':<32} {r_score_cnn:>10.4f} {ode_score:>12.2e}")
print(f"{'='*65}")
print(f"\nNote: R-score = MSE during DA ON periods only")
print(f"      ODE-score computed with RK4 (same as Fablet)")

# ── Spatial Maps ──────────────────────────────────────
TSHOW = 200
vmin  = X_true[:TSHOW].min()
vmax  = X_true[:TSHOW].max()
emax  = 6.0

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

X_obs_vis = np.full((TSHOW, K), np.nan)
for j, oi in enumerate(obs_idx):
    X_obs_vis[:, oi] = X_obs[:TSHOW, j]

data_top   = [X_true[:TSHOW], X_obs_vis,
              X_3dv[:TSHOW],  X_cnn[:TSHOW]]
titles_top = ['True State',
              'Observations\n(20/40 vars, 5% noise)',
              '3D-Var Reconstruction',
              'Cell-NN Reconstruction']

for col, (data, title) in enumerate(zip(data_top, titles_top)):
    ax = fig.add_subplot(gs[0, col])
    im = ax.imshow(data.T, aspect='auto', cmap='RdBu_r',
                   vmin=vmin, vmax=vmax, origin='lower')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Variable k')
    plt.colorbar(im, ax=ax, shrink=0.85)

err_bg      = np.abs(X_true[:TSHOW] - np.tile(X0_bg, (TSHOW,1)))
err_obs_vis = np.full((TSHOW, K), np.nan)
for j, oi in enumerate(obs_idx):
    err_obs_vis[:, oi] = np.abs(X_true[:TSHOW, oi] - X_obs[:TSHOW, j])
err_3dv_map = np.abs(X_true[:TSHOW] - X_3dv[:TSHOW])
err_cnn_map = np.abs(X_true[:TSHOW] - X_cnn[:TSHOW])

data_bot   = [err_bg, err_obs_vis, err_3dv_map, err_cnn_map]
titles_bot = ['Background Error',
              'Observation Error',
              f'3D-Var Error\nR={r_score_3dv:.3f}',
              f'Cell-NN Error\nR={r_score_cnn:.3f}']

for col, (data, title) in enumerate(zip(data_bot, titles_bot)):
    ax = fig.add_subplot(gs[1, col])
    em = 2.0 if col == 1 else emax
    im = ax.imshow(data.T, aspect='auto', cmap='hot_r',
                   vmin=0, vmax=em, origin='lower')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Variable k')
    plt.colorbar(im, ax=ax, shrink=0.85)

fig.suptitle(
    f'Lorenz-96 Spatial Maps — Cell-NN DA\n'
    f'Cell-NN R={r_score_cnn:.3f} | 3D-Var R={r_score_3dv:.3f} | '
    f'ODE-score(RK4)={ode_score:.2e}\n'
    f'αA=1.0, wb=1.0, da_interval=25, noise=5%',
    fontsize=12, fontweight='bold'
)
plt.savefig('results/spatial_maps_final.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved → results/spatial_maps_final.png")

# Save scores
np.save('results/r_score_cnn_final.npy',  np.array([r_score_cnn]))
np.save('results/r_score_3dv_final.npy',  np.array([r_score_3dv]))
np.save('results/ode_score_rk4_final.npy', np.array([ode_score]))
print("All scores saved!")
