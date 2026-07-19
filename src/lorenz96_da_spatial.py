"""
Lorenz-96 Cell-NN DA: Full-state spatial maps, DA-on vs DA-off.
=================================================================
WHAT THIS SCRIPT DOES (high level):

lorenz96_da.py's main figure (Figure 5) only shows ONE of the 40
state variables (X_1) plotted over time. That's useful for seeing
the trajectory shape, but it hides what's happening to the OTHER 39
variables, and it doesn't make the DA-on vs DA-off contrast as
visually obvious as it could be.

This script re-runs the *exact same* primary experiment (same
random seeds, same parameters, same DA-on/off schedule) but this
time saves and plots the FULL 40-variable state as a space-time
heatmap: one axis is "which of the 40 variables" (k), the other
axis is "time" (t). This lets us see, at a glance, how ALL 40
variables behave together, and how sharply the reconstruction
quality changes the moment DA switches off (and recovers the
moment it switches back on).

IMPORTANT: this script does NOT modify or re-run anything that the
original Figure 5, Table 2, or Supplementary Figure S1 depend on.
It is a completely separate, standalone script that happens to reuse
the same underlying run_cellnn_lorenz96() function from lorenz96_da.py.
This keeps all of the paper's existing, already-verified results
untouched.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from src.lorenz96 import (integrate_lorenz96, make_initial_condition,
                           generate_observations_96)
# Reuse the SAME Cell-NN DA loop function used to produce Figure 5 /
# Table 2 -- this guarantees the physics and DA logic are identical,
# we're just extracting and plotting more of the output than before.
from src.lorenz96_da import run_cellnn_lorenz96

print("="*65)
print("Lorenz-96 Cell-NN DA: Full-state spatial maps (DA on vs off)")
print("="*65)

# ── Exact same setup as lorenz96_da.py's primary realization ──
# These MUST match lorenz96_da.py exactly, otherwise we'd be looking
# at a different (not directly comparable) experiment.
K, F, dt, N = 40, 8.0, 1e-2, 5000
da_interval = 25
da_on_periods = [(0, 2000), (3500, 5000)]  # DA off during steps 2000-3500
                                              # (= t=20 to t=35)

# Same random seeds as the main experiment -- this reproduces the
# IDENTICAL true trajectory and background trajectory used for
# Figure 5 and Table 2. Nothing here is a "new" or "different" run.
X0_true = make_initial_condition(K, F, perturb=0.0, seed=42)
X0_bg   = make_initial_condition(K, F, perturb=0.05, seed=123)

print("\nGenerating true trajectory...")
t, X_true = integrate_lorenz96(X0_true, (0, N*dt), dt)

print("Generating observations (20/40 variables)...")
X_obs, obs_idx = generate_observations_96(X_true, noise_level=0.05, obs_every=2)

print("\nRunning Cell-NN DA (full state saved this time)...")
# Unlike lorenz96_da.py's main script (which only keeps track of
# summary error statistics), here we keep the FULL X_cnn array --
# every one of the 40 variables, at every one of the 5000 timesteps.
X_cnn, errors_cnn = run_cellnn_lorenz96(
    X_true, X_obs, obs_idx, X0_bg, dt,
    da_interval=da_interval, da_on_periods=da_on_periods,
    alpha_A=1.0, wb=1.0, K=K, F=F
)

print(f"X_true shape: {X_true.shape}, X_cnn shape: {X_cnn.shape}")

# Quick sanity check: this mean absolute error is NOT expected to
# exactly match Table 2's headline "0.310" number -- that number is
# the R-score (mean SQUARED error, restricted to DA-ON periods only).
# This is a plain mean ABSOLUTE error averaged over the WHOLE
# timeline, including the high-error DA-OFF window -- so it will be
# noticeably higher. That's expected, not a bug.
mean_err = np.mean(np.abs(X_true - X_cnn))
print(f"Mean abs error (sanity check): {mean_err:.4f}")

# Save the full state arrays so we (or anyone else) can re-plot or
# re-analyze this exact run later without re-simulating everything.
np.save('results/lorenz96_spatial_X_true.npy', X_true)
np.save('results/lorenz96_spatial_X_cnn.npy', X_cnn)

# ── Build the spatial (variable k vs time) comparison figure ──
# Convert the DA-off step-boundaries into real time units (dt=0.01),
# matching exactly what Figure 5's caption already states (t=20-35).
t_off_start, t_off_end = 2000*dt, 3500*dt  # = 20, 35

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Three panels:
#   (a) the TRUE state, as a k-vs-t heatmap
#   (b) the Cell-NN RECONSTRUCTION, same layout
#   (c) the ABSOLUTE ERROR between them, |true - reconstruction|
panels = [
    (X_true, 'True State $X_{true}$'),
    (X_cnn,  'Cell-NN Reconstruction $\\hat{X}$'),
    (np.abs(X_true - X_cnn), 'Absolute Error $|X_{true} - \\hat{X}|$'),
]

# Colorbar range for panels (a) and (b): rather than using a fixed
# guess like -10/10, we compute the actual 1st-to-99th percentile
# range of the real data. This keeps the color scale well-matched to
# the real spread of values -- wide enough to avoid clipping real
# signal, tight enough that the color variation is still informative
# (a plain min/max would let a handful of extreme outlier values
# wash out the color scale for everything else).
vmin_state = min(np.percentile(X_true, 1), np.percentile(X_cnn, 1))
vmax_state = max(np.percentile(X_true, 99), np.percentile(X_cnn, 99))
print(f'State colorbar range (1st-99th pct): {vmin_state:.2f} to {vmax_state:.2f}')
panel_letters = ['(a)', '(b)', '(c)']

for i, (data, title) in enumerate(panels):
    ax = axes[i]
    # Panels (a),(b): diverging red/blue colormap for the raw state
    #   values (can be positive or negative).
    # Panel (c): "hot" colormap for the error (always >= 0, so we
    #   want a sequential, not diverging, color scheme).
    cmap = 'RdBu_r' if i < 2 else 'hot_r'
    vmin, vmax = (vmin_state, vmax_state) if i < 2 else (0, 5)
    im = ax.pcolormesh(np.arange(K), t, data, cmap=cmap,
                        vmin=vmin, vmax=vmax, shading='auto')

    # Two horizontal dashed lines mark exactly where DA switches off
    # (t=20) and back on (t=35). Drawn in solid black for visibility
    # (an earlier neon-green version was too hard to see).
    ax.axhline(t_off_start, color='black', lw=2.0, linestyle='--',
               label='DA-off boundary' if i == 0 else None)
    ax.axhline(t_off_end, color='black', lw=2.0, linestyle='--')

    # Text labels along the right edge of each panel, marking which
    # of the three time regions (DA ON / DA OFF / DA ON) is which --
    # placed outside the plotted data so they don't obscure anything.
    ax.text(K*1.02, t_off_start*0.5, 'DA ON', ha='left', va='center',
            fontsize=9, fontweight='bold', color='black', rotation=90)
    ax.text(K*1.02, (t_off_start+t_off_end)/2, 'DA OFF', ha='left', va='center',
            fontsize=9, fontweight='bold', color='black', rotation=90)
    ax.text(K*1.02, (t_off_end+t.max())/2, 'DA ON', ha='left', va='center',
            fontsize=9, fontweight='bold', color='black', rotation=90)

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Variable index $k$')
    if i == 0:
        ax.set_ylabel('Time')
    # Panel letter (a)/(b)/(c) in the top-left corner of each panel
    ax.text(0.02, 0.98, panel_letters[i], transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top', color='black',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.colorbar(im, ax=ax, shrink=0.8)

axes[0].legend(loc='upper right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('results/lorenz96_spatial_da_onoff.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved -> results/lorenz96_spatial_da_onoff.png")

# ── Quantify DA-on vs DA-off performance spatially ──
# Build a boolean mask marking which of the 5000 timesteps fall
# inside a DA-ON period, vs which fall in the DA-OFF gap.
on_mask = np.zeros(N, dtype=bool)
for s, e in da_on_periods:
    on_mask[s:e] = True
off_mask = ~on_mask

# Compute the average absolute error separately for DA-on steps and
# DA-off steps. This gives a single, clean headline number
# summarizing exactly how much worse the reconstruction gets once
# observations stop arriving.
err_on = np.mean(np.abs(X_true[on_mask] - X_cnn[on_mask]))
err_off = np.mean(np.abs(X_true[off_mask] - X_cnn[off_mask]))
print(f"\nMean abs error, DA-ON steps : {err_on:.4f}")
print(f"Mean abs error, DA-OFF steps: {err_off:.4f}")
print(f"Ratio (off/on): {err_off/err_on:.2f}x")