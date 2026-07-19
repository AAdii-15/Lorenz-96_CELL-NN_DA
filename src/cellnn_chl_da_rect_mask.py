"""
Cell-NN DA on MODIS Chl-a — RECTANGULAR COASTAL MASK VERSION
================================================================
Same model as cellnn_chl_da_proper.py (identical cellnn_da_2d),
but replaces the random 20% scattered test mask with a single
contiguous rectangular coastal region, for one representative date
(2017-01-01, same date as the original Figure 7).

Mask region: rows 80-139, cols 300-359
  lat 16.19-18.65N, lon 92.52-94.98E (Myanmar coast)
  2804 ocean pixels, ~22% land, 100% valid on 2017-01-01

Ground truth for error panel: MODIS's own true value (not L4),
per explicit instruction -- keeps this figure internally consistent
with the original Figure 7's MODIS-self-validation approach.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src.da_cellnn import cellnn_da_step

print("="*65)
print("Cell-NN DA — Rectangular Coastal Mask")
print("="*65)

# ── Load data (identical to cellnn_chl_da_proper.py) ──
chl_norm = np.load('data/modis_chl/chl_norm.npy')
dates    = np.load('data/modis_chl/chl_dates.npy')
lat      = np.load('data/modis_chl/chl_lat.npy')
lon      = np.load('data/modis_chl/chl_lon.npy')

target_idx = [i for i, d in enumerate(dates) if str(d) == '20170101'][0]
print(f"Target date: {dates[target_idx]} (composite index {target_idx})")

# ── Identical cellnn_da_2d function ────────────────────
def cellnn_da_2d(mu_b_2d, y_obs_2d, obs_mask_2d,
                 alpha_A=1.0, wb=1.0,
                 kappa=0.4, n_diff=50):
    lat_n, lon_n = mu_b_2d.shape
    mu_b_flat  = mu_b_2d.flatten()
    y_obs_flat = y_obs_2d.flatten()
    obs_flat   = obs_mask_2d.flatten()

    mu_b_obs = mu_b_flat[obs_flat]
    y_obs    = y_obs_flat[obs_flat]

    mu_a_obs, n_iter = cellnn_da_step(
        mu_b_obs, y_obs,
        alpha_A=alpha_A, wb=wb,
        d_tau=5e-2, tau_max=1e-1,
        eps_conv=1e-6, r_max=50
    )

    mu_a_flat = mu_b_flat.copy()
    mu_a_flat[obs_flat] = mu_a_obs
    mu_a_2d = mu_a_flat.reshape(lat_n, lon_n)

    for _ in range(n_diff):
        mu_pad = np.pad(mu_a_2d, 1, mode='edge')
        laplacian = (mu_pad[:-2, 1:-1] + mu_pad[2:, 1:-1] +
                     mu_pad[1:-1, :-2] + mu_pad[1:-1, 2:] -
                     4.0 * mu_a_2d)
        update = np.zeros_like(mu_a_2d)
        update[~obs_mask_2d] = kappa * laplacian[~obs_mask_2d]
        mu_a_2d = mu_a_2d + 0.1 * update

    return mu_a_2d, n_iter

# ── Set up rectangular coastal mask ────────────────────
r0, c0, box_size = 80, 300, 60
r1, c1 = r0 + box_size, c0 + box_size
print(f"Mask region: rows {r0}-{r1}, cols {c0}-{c1}")
print(f"  lat {lat[r1-1]:.2f}-{lat[r0]:.2f}N, lon {lon[c0]:.2f}-{lon[c1-1]:.2f}E")

field_curr = chl_norm[target_idx]
field_prev = chl_norm[target_idx - 1]
valid_mask = ~np.isnan(field_curr)

mu_b = field_prev.copy()
mu_b[np.isnan(mu_b)] = 0.0

# Rectangular test region: only ocean pixels within the box
rect_region = np.zeros_like(valid_mask)
rect_region[r0:r1, c0:c1] = True
test_mask = rect_region & valid_mask   # ocean pixels within box, valid on this date

n_test = test_mask.sum()
print(f"Ocean pixels withheld in rectangle: {n_test}")

obs_mask = valid_mask.copy()
obs_mask[test_mask] = False

y_obs = np.where(obs_mask, field_curr, 0.0)

print("\nRunning Cell-NN DA...")
mu_a, n_iter = cellnn_da_2d(mu_b, y_obs, obs_mask,
                             alpha_A=1.0, wb=1.0, kappa=0.4, n_diff=50)

true_v = field_curr[test_mask]
an_v   = mu_a[test_mask]
bg_v   = mu_b[test_mask]

rmse_an = np.sqrt(np.mean((true_v - an_v)**2))
rmse_b  = np.sqrt(np.mean((true_v - bg_v)**2))
imp = (rmse_b - rmse_an) / rmse_b * 100

print(f"\n{'='*65}")
print(f"RESULTS (rectangular coastal mask, n={n_test} pixels)")
print(f"{'='*65}")
print(f"Background RMSE : {rmse_b:.4f}")
print(f"Cell-NN RMSE    : {rmse_an:.4f}")
print(f"Improvement     : {imp:.1f}%")
print(f"Iterations      : {n_iter}")
print(f"{'='*65}")

# ── 6-panel spatial figure (same structure as Figure 7) ──
natural_gap = np.isnan(field_curr)
test_hidden = test_mask

category_map = np.full(field_curr.shape, np.nan)
category_map[natural_gap] = 0
category_map[obs_mask]    = 1
category_map[test_hidden] = 2

cmap_mask = ListedColormap(['white', '#a8d0e6', '#e63946'])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm_mask = BoundaryNorm(bounds, cmap_mask.N)

panels = [
    (field_curr, 'MODIS L3', 'RdBu_r', -3, 3),
    (np.where(obs_mask, field_curr, np.nan),
     'Observation at Time t\nX(t)', 'RdBu_r', -3, 3),
    (mu_b, 'Background\nX(t-1)', 'RdBu_r', -3, 3),
    (mu_a, r'Cell-NN Reconstruction' + '\n' + r'$\hat{X}(t)$', 'RdBu_r', -3, 3),
    (np.abs(field_curr - mu_a),
     r'Absolute Error' + '\n' + r'$|X(t) - \hat{X}(t)|$', 'hot_r', 0, 0.3),
]

fig = plt.figure(figsize=(24, 5))
gs = gridspec.GridSpec(1, 6, wspace=0.65)

panel_letters = ['(a)', '(b)', '(c)', '(d)', '(e)']
for col, (data, title, cmap, vn, vx) in enumerate(panels):
    ax = fig.add_subplot(gs[0, col])
    im = ax.imshow(data,
        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        origin='upper', cmap=cmap, vmin=vn, vmax=vx, aspect='auto')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.text(0.02, 0.98, panel_letters[col], transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', color='black')
    if col == 0:
        ax.set_ylabel('Latitude', fontsize=11, labelpad=18)
        ax.yaxis.set_label_coords(-0.28, 0.5)
    plt.colorbar(im, ax=ax, shrink=0.8)

ax6 = fig.add_subplot(gs[0, 5])
im6 = ax6.imshow(category_map,
    extent=[lon.min(), lon.max(), lat.min(), lat.max()],
    origin='upper', cmap=cmap_mask, norm=norm_mask, aspect='auto')
ax6.set_title('Test Mask\n(Rectangular, Coastal)', fontsize=10, fontweight='bold')
ax6.text(0.02, 0.98, '(f)', transform=ax6.transAxes,
         fontsize=12, fontweight='bold', va='top', color='black')
cbar6 = plt.colorbar(im6, ax=ax6, shrink=0.8, ticks=[0, 1, 2])
cbar6.ax.set_yticklabels(['Natural\ncloud gap', 'Input to\nmodel', 'Held-out\n(rectangle)'],
                         fontsize=7)

fig.text(0.5, 0.02, 'Longitude', ha='center', fontsize=10)
plt.savefig('results/chl_rect_mask_spatial.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved -> results/chl_rect_mask_spatial.png")
