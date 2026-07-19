"""
Cell-NN DA — Fixed SE-corner rectangular mask, L4 ground truth.
================================================================
Polished version:
  - Absolute Error panel restored (vs L4), tight colorbar range
  - Redundant "MODIS L3 (unmasked)" panel dropped
  - Shared colorbar across the four RdBu_r state panels
  - Locator inset showing the zoom region within the full domain
- Fixed region (from paper's own text): SE corner near Andaman Sea,
  rows 182-271, cols 427-480 (10.7-14.4N, 97.8-100E), 1833 ocean
  pixels, 61% land. Coordinates never change.
- Background = real previous-composite MODIS field X(t-1).
- Ground truth for scoring = L4 (stable, gap-free reference).
- Single target date: 2017-01-01.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src.da_cellnn import cellnn_da_step

print("="*65)
print("Cell-NN DA -- Fixed SE-corner mask, MODIS background, L4 truth")
print("="*65)

chl_norm = np.load('data/modis_chl/chl_norm.npy')
dates    = np.load('data/modis_chl/chl_dates.npy')
lat      = np.load('data/modis_chl/chl_lat.npy')
lon      = np.load('data/modis_chl/chl_lon.npy')
l4_norm  = np.load('data/l4_chl/chl_l4_norm.npy')

target_idx = [i for i, d in enumerate(dates) if str(d) == '20170101'][0]
print(f"Target date: {dates[target_idx]} (composite index {target_idx})")

r0, r1 = 182, 271
c0, c1 = 427, 480
print(f"Mask region (fixed): rows {r0}-{r1}, cols {c0}-{c1}")
print(f"  lat {lat[r1-1]:.2f}-{lat[r0]:.2f}N, lon {lon[c0]:.2f}-{lon[c1-1]:.2f}E")

def cellnn_da_2d(mu_b_2d, y_obs_2d, obs_mask_2d,
                 alpha_A=1.0, wb=1.0, kappa=0.4, n_diff=50):
    lat_n, lon_n = mu_b_2d.shape
    mu_b_flat = mu_b_2d.flatten(); y_obs_flat = y_obs_2d.flatten(); obs_flat = obs_mask_2d.flatten()
    mu_b_obs = mu_b_flat[obs_flat]; y_obs = y_obs_flat[obs_flat]
    mu_a_obs, n_iter = cellnn_da_step(
        mu_b_obs, y_obs, alpha_A=alpha_A, wb=wb,
        d_tau=5e-2, tau_max=1e-1, eps_conv=1e-6, r_max=50)
    mu_a_flat = mu_b_flat.copy(); mu_a_flat[obs_flat] = mu_a_obs
    mu_a_2d = mu_a_flat.reshape(lat_n, lon_n)
    for _ in range(n_diff):
        mu_pad = np.pad(mu_a_2d, 1, mode='edge')
        laplacian = (mu_pad[:-2,1:-1]+mu_pad[2:,1:-1]+mu_pad[1:-1,:-2]+mu_pad[1:-1,2:]-4.0*mu_a_2d)
        update = np.zeros_like(mu_a_2d); update[~obs_mask_2d] = kappa*laplacian[~obs_mask_2d]
        mu_a_2d = mu_a_2d + 0.1*update
    return mu_a_2d, n_iter

field_curr = chl_norm[target_idx]
field_prev = chl_norm[target_idx - 1]
l4_field   = l4_norm[target_idx]
valid_mask = ~np.isnan(field_curr)

rect_region = np.zeros_like(valid_mask)
rect_region[r0:r1, c0:c1] = True
test_mask = rect_region & valid_mask
n_test = test_mask.sum()
print(f"Ocean pixels withheld in fixed rectangle: {n_test}")

obs_mask = valid_mask.copy()
obs_mask[test_mask] = False

mu_b = field_prev.copy()
mu_b[np.isnan(mu_b)] = 0.0
y_obs = np.where(obs_mask, field_curr, 0.0)

print("\nRunning Cell-NN DA (real MODIS background)...")
mu_a, n_iter = cellnn_da_2d(mu_b, y_obs, obs_mask, alpha_A=1.0, wb=1.0, kappa=0.4, n_diff=50)

l4_valid_in_mask = ~np.isnan(l4_field) & test_mask
n_scored = l4_valid_in_mask.sum()
true_v = l4_field[l4_valid_in_mask]
an_v   = mu_a[l4_valid_in_mask]
bg_v   = mu_b[l4_valid_in_mask]

rmse_an = np.sqrt(np.mean((true_v - an_v)**2))
rmse_bg = np.sqrt(np.mean((true_v - bg_v)**2))
imp = (rmse_bg - rmse_an) / rmse_bg * 100

print(f"\n{'='*65}")
print(f"RESULTS (fixed SE-corner mask, MODIS background, scored vs L4, n={n_scored})")
print(f"{'='*65}")
print(f"Background RMSE vs L4 : {rmse_bg:.4f}")
print(f"Cell-NN RMSE vs L4    : {rmse_an:.4f}")
print(f"Improvement           : {imp:+.1f}%")
print(f"Iterations             : {n_iter}")
print(f"{'='*65}")

# Data-driven error colorbar range (90th percentile within the masked region)
error_field = np.abs(l4_field - mu_a)
err_vmax = np.percentile(error_field[l4_valid_in_mask], 90)
print(f"Error colorbar vmax (90th pct within mask): {err_vmax:.3f}")

# ── Figure geometry ──
rect_lon0, rect_lon1 = lon[c0], lon[c1-1]
rect_lat0, rect_lat1 = lat[r1-1], lat[r0]
pad_lat, pad_lon = 3.0, 4.0
zoom_lat0 = max(lat.min(), rect_lat0 - pad_lat)
zoom_lat1 = min(lat.max(), rect_lat1 + pad_lat)
zoom_lon0 = max(lon.min(), rect_lon0 - pad_lon)
zoom_lon1 = min(lon.max(), rect_lon1 + pad_lon)

natural_gap = np.isnan(field_curr)
category_map = np.full(field_curr.shape, np.nan)
category_map[natural_gap] = 0
category_map[obs_mask]    = 1
category_map[test_mask]   = 2
cmap_mask = ListedColormap(['white', '#a8d0e6', '#e63946'])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm_mask = BoundaryNorm(bounds, cmap_mask.N)

# Panels: Observation, Background, L4 truth, Cell-NN recon (shared cbar),
# then Absolute Error (own cbar), then categorical mask (own cbar)
state_panels = [
    (np.where(obs_mask, field_curr, np.nan), 'Observation at Time t\nX(t)'),
    (mu_b, 'Background\nX(t-1)'),
    (l4_field, 'L4 Ground Truth\n(fixed reference)'),
    (mu_a, r'Cell-NN Reconstruction' + '\n' + r'$\hat{X}(t)$'),
]

fig = plt.figure(figsize=(24, 6))
gs = gridspec.GridSpec(1, 6, width_ratios=[1,1,1,1,1,1.15], wspace=0.35)
panel_letters = ['(a)', '(b)', '(c)', '(d)']

axes_state = []
for col, (data, title) in enumerate(state_panels):
    ax = fig.add_subplot(gs[0, col])
    im = ax.imshow(data, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                    origin='upper', cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
    ax.set_xlim(zoom_lon0, zoom_lon1); ax.set_ylim(zoom_lat0, zoom_lat1)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.text(0.03, 0.97, panel_letters[col], transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')
    if col > 0:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel('Latitude', fontsize=10)
    ax.set_xlabel('Longitude', fontsize=9)
    if col == 0:
        # Solid highlight only on the Observation panel, where the
        # mask genuinely applies (data is hidden from the model here)
        rect = patches.Rectangle((rect_lon0, rect_lat0), rect_lon1-rect_lon0, rect_lat1-rect_lat0,
                                  linewidth=2, edgecolor='#00ff00', facecolor='#00ff00', alpha=0.30)
    else:
        # Neutral outline elsewhere -- marks the region for visual
        # alignment only; nothing was hidden/masked in these panels
        rect = patches.Rectangle((rect_lon0, rect_lat0), rect_lon1-rect_lon0, rect_lat1-rect_lat0,
                                  linewidth=1.5, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    axes_state.append(ax)

# Shared colorbar for the 4 state panels -- standard robust pattern,
# attached to the list of axes rather than a manually placed GridSpec cell
cb_shared = fig.colorbar(im, ax=axes_state, shrink=0.85, pad=0.015, aspect=25)
cb_shared.set_label('log$_{10}$(Chl-a), normalized', fontsize=9)

# Absolute error panel -- separate axis, own colorbar, plenty of pad
# so it never collides with the shared colorbar above
ax_err = fig.add_subplot(gs[0, 4])
im_err = ax_err.imshow(error_field, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                        origin='upper', cmap='hot_r', vmin=0, vmax=err_vmax, aspect='auto')
ax_err.set_xlim(zoom_lon0, zoom_lon1); ax_err.set_ylim(zoom_lat0, zoom_lat1)
ax_err.set_title(r'Absolute Error vs L4' + '\n' + r'$|L4 - \hat{X}(t)|$', fontsize=10, fontweight='bold')
ax_err.text(0.03, 0.97, '(e)', transform=ax_err.transAxes, fontsize=12, fontweight='bold', va='top')
ax_err.set_yticklabels([])
ax_err.set_xlabel('Longitude', fontsize=9)
rect_err = patches.Rectangle((rect_lon0, rect_lat0), rect_lon1-rect_lon0, rect_lat1-rect_lat0,
                              linewidth=1.5, edgecolor='black', facecolor='none')
ax_err.add_patch(rect_err)
cb_err = fig.colorbar(im_err, ax=ax_err, shrink=0.85, pad=0.03)
cb_err.set_label('|error|', fontsize=9)

# Categorical mask panel
ax6 = fig.add_subplot(gs[0, 5])
im6 = ax6.imshow(category_map, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                  origin='upper', cmap=cmap_mask, norm=norm_mask, aspect='auto')
ax6.set_xlim(zoom_lon0, zoom_lon1); ax6.set_ylim(zoom_lat0, zoom_lat1)
ax6.set_title('Fixed Test Mask\n(SE Corner, Constant)', fontsize=10, fontweight='bold')
ax6.text(0.03, 0.97, '(f)', transform=ax6.transAxes, fontsize=12, fontweight='bold', va='top')
ax6.set_yticklabels([])
ax6.set_xlabel('Longitude', fontsize=9)
rect6 = patches.Rectangle((rect_lon0, rect_lat0), rect_lon1-rect_lon0, rect_lat1-rect_lat0,
                           linewidth=2, edgecolor='black', facecolor='none')
ax6.add_patch(rect6)
cbar6 = plt.colorbar(im6, ax=ax6, shrink=0.85, ticks=[0, 1, 2])
cbar6.ax.set_yticklabels(['Natural\ncloud gap', 'Input to\nmodel', 'Held-out\n(fixed mask)'], fontsize=7)

# Locator inset on panel (a): full domain thumbnail with red zoom box
inset_ax = axes_state[0].inset_axes([0.02, 0.02, 0.38, 0.38])
inset_ax.imshow(field_curr, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                 origin='upper', cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
inset_ax.add_patch(patches.Rectangle((zoom_lon0, zoom_lat0), zoom_lon1-zoom_lon0, zoom_lat1-zoom_lat0,
                                      linewidth=1, edgecolor='red', facecolor='none'))
inset_ax.set_xticks([]); inset_ax.set_yticks([])
for spine in inset_ax.spines.values():
    spine.set_edgecolor('black'); spine.set_linewidth(1)

plt.savefig('results/chl_rect_mask_l4_spatial.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved -> results/chl_rect_mask_l4_spatial.png")
