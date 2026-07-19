"""
Cell-NN DA on MODIS Chl-a — SYNCED WITH OUR MODEL
===================================================
Uses EXACT same cellnn_da_step() from da_cellnn.py
Same parameters: alpha_A=1.0, wb=1.0

Framework:
  Background : X(t-1) — previous 8-day image
  Observations: Y(t)  — current satellite pixels
  
  For each pixel:
    Flatten 2D → cellnn_da_step() → Reshape 2D
    
  Unobserved pixels: spatial diffusion
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter
from datetime import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# ── Import OUR Cell-NN DA model ───────────────────────
from src.da_cellnn import cellnn_da_step

print("="*65)
print("Cell-NN DA on MODIS Chl-a")
print("Using OUR cellnn_da_step() from da_cellnn.py")
print("Parameters: alpha_A=1.0, wb=1.0 (optimized)")
print("="*65)

# ── Load data ─────────────────────────────────────────
chl_norm = np.load('data/modis_chl/chl_norm.npy')
dates    = np.load('data/modis_chl/chl_dates.npy')
lat      = np.load('data/modis_chl/chl_lat.npy')
lon      = np.load('data/modis_chl/chl_lon.npy')
print(f"Shape : {chl_norm.shape}")
print(f"Period: {dates[0]} to {dates[-1]}")

# ── Our Cell-NN DA for 2D field ───────────────────────
def cellnn_da_2d(mu_b_2d, y_obs_2d, obs_mask_2d,
                 alpha_A=1.0, wb=1.0,
                 kappa=0.4, n_diff=50):
    """
    Apply our cellnn_da_step() to 2D spatial field.

    Steps:
    1. Flatten 2D → 1D
    2. Apply cellnn_da_step() on observed pixels
    3. Reshape back to 2D
    4. Fill unobserved gaps with spatial diffusion

    Parameters
    ----------
    mu_b_2d   : (lat,lon) — background = X(t-1)
    y_obs_2d  : (lat,lon) — observations Y(t)
    obs_mask_2d:(lat,lon) bool — True=observed
    alpha_A   : float — our optimized value = 1.0
    wb        : float — our optimized value = 1.0
    kappa     : float — diffusion for gap filling
    n_diff    : int   — diffusion iterations

    Returns
    -------
    mu_a_2d : (lat,lon) — analysis field
    """
    lat_n, lon_n = mu_b_2d.shape

    # ── Step 1: Flatten ───────────────────────────────
    mu_b_flat    = mu_b_2d.flatten()       # (lat*lon,)
    y_obs_flat   = y_obs_2d.flatten()      # (lat*lon,)
    obs_flat     = obs_mask_2d.flatten()   # (lat*lon,)

    # ── Step 2: Apply cellnn_da_step on obs pixels ────
    # Extract only observed pixels
    mu_b_obs = mu_b_flat[obs_flat]         # (n_obs,)
    y_obs    = y_obs_flat[obs_flat]        # (n_obs,)

    # Apply OUR Cell-NN DA!
    mu_a_obs, n_iter = cellnn_da_step(
        mu_b_obs, y_obs,
        alpha_A=alpha_A,
        wb=wb,
        d_tau=5e-2,
        tau_max=1e-1,
        eps_conv=1e-6,
        r_max=50
    )

    # ── Step 3: Put back into 2D ──────────────────────
    mu_a_flat = mu_b_flat.copy()
    mu_a_flat[obs_flat] = mu_a_obs
    mu_a_2d   = mu_a_flat.reshape(lat_n, lon_n)

    # ── Step 4: Fill gaps with spatial diffusion ──────
    # Unobserved pixels: diffuse from neighbors
    # (background structure + observed neighbors)
    for _ in range(n_diff):
        mu_pad    = np.pad(mu_a_2d, 1, mode='edge')
        laplacian = (mu_pad[:-2, 1:-1] +
                     mu_pad[2:,  1:-1] +
                     mu_pad[1:-1, :-2] +
                     mu_pad[1:-1, 2:] -
                     4.0 * mu_a_2d)

        # Only update unobserved pixels
        update              = np.zeros_like(mu_a_2d)
        update[~obs_mask_2d] = kappa * laplacian[~obs_mask_2d]
        mu_a_2d             = mu_a_2d + 0.1 * update

    return mu_a_2d, n_iter

# ── Full Experiment ───────────────────────────────────
print("\nExperiment:")
print("  Period     : 2015-2019 (all 5 years)")
print("  Background : X(t-1) previous 8-day image")
print("  Test mask  : 20% valid pixels hidden")
print("  Model      : cellnn_da_step() — same as L96!")
print("  Params     : alpha_A=1.0, wb=1.0")

np.random.seed(42)

rmse_cellnn = []
rmse_bg     = []
dates_used  = []
n_iters     = []
saved_maps  = []

print("\nRunning...")

for t in range(1, len(chl_norm)):

    field_curr = chl_norm[t]
    field_prev = chl_norm[t-1]
    valid_mask = ~np.isnan(field_curr)

    if valid_mask.sum() < 500:
        continue

    # Background = previous image
    mu_b = field_prev.copy()
    mu_b[np.isnan(mu_b)] = 0.0

    # Test mask — hide 20% valid pixels
    valid_idx = np.where(valid_mask)
    n_valid   = len(valid_idx[0])
    n_test    = int(0.2 * n_valid)
    sel       = np.random.choice(n_valid, n_test,
                                 replace=False)
    test_rows = valid_idx[0][sel]
    test_cols = valid_idx[1][sel]

    obs_mask  = valid_mask.copy()
    obs_mask[test_rows, test_cols] = False

    # Observations
    y_obs = np.where(obs_mask, field_curr, 0.0)

    # ── Apply OUR Cell-NN DA ──────────────────────────
    mu_a, n_iter = cellnn_da_2d(
        mu_b, y_obs, obs_mask,
        alpha_A=1.0, wb=1.0,
        kappa=0.4, n_diff=50
    )

    # RMSE on test pixels
    true_v = field_curr[test_rows, test_cols]
    an_v   = mu_a[test_rows, test_cols]
    bg_v   = mu_b[test_rows, test_cols]

    rmse_an = np.sqrt(np.mean((true_v - an_v)**2))
    rmse_b  = np.sqrt(np.mean((true_v - bg_v)**2))

    rmse_cellnn.append(rmse_an)
    rmse_bg.append(rmse_b)
    dates_used.append(dates[t])
    n_iters.append(n_iter)

    if t % 23 == 0:
        saved_maps.append({
            'date'  : dates[t],
            'true'  : field_curr,
            'bg'    : mu_b,
            'an'    : mu_a,
            'obs'   : obs_mask,
            'rmse_an': rmse_an,
            'rmse_bg': rmse_b,
        })

    if t % 46 == 0:
        imp = (rmse_b-rmse_an)/rmse_b*100
        print(f"  t={t:3d} | {dates[t]} | "
              f"BG={rmse_b:.4f} | "
              f"CNN={rmse_an:.4f} | "
              f"Imp={imp:.1f}% | "
              f"iters={n_iter}")

# ── Results ───────────────────────────────────────────
rmse_cellnn = np.array(rmse_cellnn)
rmse_bg     = np.array(rmse_bg)
improvement = ((rmse_bg.mean()-rmse_cellnn.mean())
               /rmse_bg.mean()*100)

print(f"\n{'='*65}")
print(f"FINAL RESULTS")
print(f"{'='*65}")
print(f"{'Metric':<30} {'Cell-NN':>12} {'BG(t-1)':>12}")
print(f"{'-'*65}")
print(f"{'Mean RMSE':<30} {rmse_cellnn.mean():>12.4f} "
      f"{rmse_bg.mean():>12.4f}")
print(f"{'Std RMSE':<30} {rmse_cellnn.std():>12.4f} "
      f"{rmse_bg.std():>12.4f}")
print(f"{'Min RMSE':<30} {rmse_cellnn.min():>12.4f} "
      f"{rmse_bg.min():>12.4f}")
print(f"{'Max RMSE':<30} {rmse_cellnn.max():>12.4f} "
      f"{rmse_bg.max():>12.4f}")
print(f"{'Mean DA iterations':<30} "
      f"{np.mean(n_iters):>12.1f}")
print(f"{'Improvement':<30} {improvement:>11.1f}%")
print(f"{'='*65}")

np.save('results/chl_proper_rmse_cellnn.npy', rmse_cellnn)
np.save('results/chl_proper_rmse_bg.npy',     rmse_bg)

# ── Plot 1: RMSE over time ─────────────────────────────
date_objs = [datetime.strptime(d, '%Y%m%d')
             for d in dates_used]

fig, axes = plt.subplots(2, 1, figsize=(16, 10))

axes[0].plot(date_objs, rmse_bg,
             'r-', lw=1.2,
             label=f'Background(t-1) '
                   f'RMSE={rmse_bg.mean():.4f}')
axes[0].plot(date_objs, rmse_cellnn,
             'b-', lw=1.2,
             label=f'Cell-NN '
                   f'RMSE={rmse_cellnn.mean():.4f}')
axes[0].fill_between(date_objs,
                     rmse_bg, rmse_cellnn,
                     where=rmse_bg>rmse_cellnn,
                     alpha=0.3, color='green',
                     label=f'Imp={improvement:.1f}%')
axes[0].set_title(
    'Cell-NN DA — MODIS Chl-a (2015-2019)\n'
    'Our cellnn_da_step() | '
    'Background=X(t-1) | αA=1.0, wb=1.0',
    fontsize=12, fontweight='bold')
axes[0].set_ylabel('RMSE (normalized)')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

imp_ts = (rmse_bg-rmse_cellnn)/rmse_bg*100
axes[1].plot(date_objs, imp_ts,
             'g-', lw=1.0, alpha=0.7)
axes[1].axhline(y=imp_ts.mean(),
                color='darkgreen', lw=2,
                linestyle='--',
                label=f'Mean={imp_ts.mean():.1f}%')
axes[1].fill_between(date_objs, 0, imp_ts,
                     alpha=0.3, color='green')
axes[1].set_title('Improvement % over Time',
                  fontsize=12)
axes[1].set_ylabel('Improvement %')
axes[1].set_xlabel('Date')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/chl_proper_rmse.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved → results/chl_proper_rmse.png")

# ── Plot 2: Spatial Maps ───────────────────────────────
m = saved_maps[3]

fig = plt.figure(figsize=(24, 5))
gs  = gridspec.GridSpec(1, 6, wspace=0.65)

# reconstruct which pixels were natural gap vs. artificially hidden
natural_gap = np.isnan(m['true'])
test_hidden = (~natural_gap) & (~m['obs'])

category_map = np.full(m['true'].shape, np.nan)
category_map[natural_gap] = 0
category_map[m['obs']]    = 1
category_map[test_hidden] = 2

from matplotlib.colors import ListedColormap, BoundaryNorm
cmap_mask = ListedColormap(['white', '#a8d0e6', '#e63946'])
bounds    = [-0.5, 0.5, 1.5, 2.5]
norm_mask = BoundaryNorm(bounds, cmap_mask.N)

panels = [
    (m['true'],
     'MODIS L3',
     'RdBu_r', -3, 3),
    (np.where(m['obs'], m['true'], np.nan),
     'Observation at Time t\nX(t)',
     'RdBu_r', -3, 3),
    (m['bg'],
     'Background\nX(t-1)',
     'RdBu_r', -3, 3),
    (m['an'],
     r'Cell-NN Reconstruction' + '\n' + r'$\hat{X}(t)$',
     'RdBu_r', -3, 3),
    (np.abs(m['true']-m['an']),
     r'Absolute Error' + '\n' + r'$|X(t) - \hat{X}(t)|$',
     'hot_r', 0, 0.3),
]

panel_letters = ['(a)', '(b)', '(c)', '(d)', '(e)']
for col, (data, title, cmap, vn, vx) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col])
    im = ax.imshow(data,
        extent=[lon.min(),lon.max(),
                lat.min(),lat.max()],
        origin='upper', cmap=cmap,
        vmin=vn, vmax=vx, aspect='auto')
    ax.set_title(title, fontsize=10,
                 fontweight='bold')
    ax.text(0.02, 0.98, panel_letters[col], transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', color='black')
    if col == 0:
        ax.set_ylabel('Latitude', fontsize=11, labelpad=18)
        ax.yaxis.set_label_coords(-0.28, 0.5)
    plt.colorbar(im, ax=ax, shrink=0.8)

ax6 = fig.add_subplot(gs[0, 5])
im6 = ax6.imshow(category_map,
    extent=[lon.min(),lon.max(), lat.min(),lat.max()],
    origin='upper', cmap=cmap_mask, norm=norm_mask, aspect='auto')
ax6.set_title('Test Mask\n(20% Withheld)',
              fontsize=10, fontweight='bold')
ax6.text(0.02, 0.98, '(f)', transform=ax6.transAxes,
         fontsize=12, fontweight='bold', va='top', color='black')

cbar6 = plt.colorbar(im6, ax=ax6, shrink=0.8, ticks=[0, 1, 2])
cbar6.ax.set_yticklabels(['Natural\ncloud gap', 'Input to\nmodel', 'Held-out\n(test set)'],
                         fontsize=7)

fig.text(0.5, 0.02, 'Longitude', ha='center', fontsize=10)
plt.savefig('results/chl_proper_spatial.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved → results/chl_proper_spatial.png")

print(f"\n{'='*65}")
print("PAPER SUMMARY")
print(f"{'='*65}")
print(f"Model      : Cell-NN DA (same as Lorenz-96!)")
print(f"Function   : cellnn_da_step() from da_cellnn.py")
print(f"Params     : αA=1.0, wb=1.0 (grid search opt.)")
print(f"Background : X(t-1) previous 8-day image")
print(f"Dataset    : MODIS Aqua Chl-a 2015-2019")
print(f"Region     : Bay of Bengal 80-100°E 5-22°N")
print(f"BG RMSE    : {rmse_bg.mean():.4f}")
print(f"CNN RMSE   : {rmse_cellnn.mean():.4f}")
print(f"Improvement: {improvement:.1f}%")
print(f"{'='*65}")
