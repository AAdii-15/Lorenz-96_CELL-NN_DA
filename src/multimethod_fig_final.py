"""
Multi-method comparison figure -- MERGED, final layout
==========================================================================
Changes from the previous merged version, per Mam's final figure pass:
  1. Ground truth (L4) moved to row 2, right after Mask.
  2. Column headers show (a)/(b)/(c)/(d) + human-readable date +
     coverage, shown once per column (top row only), not repeated as
     a title on every panel.
  3. Row labels (Mask, Filled ground truth (L4), OI, DINEOF, ...)
     shown once per row via a shared y-axis label on the leftmost
     panel, not repeated as a title on every panel in that row.
  4. Latitude and Longitude axis labels shown once for the whole
     figure (not once per panel), via fig.text().
  5. Colorbar moved to a single vertical bar on the right edge of the
     figure, instead of a horizontal bar at the bottom.
  6. Dates formatted as "August 29, 2019" instead of "20190829".
  7. Mask row explicitly labeled as MODIS L3, since that is the
     product the mask is derived from.
Compositing logic (plot_3layer/plot_2layer), data loading, and the L4
filled-ground-truth treatment are unchanged from the prior version.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datetime import datetime

print("="*65)
print("Multi-method figure -- MERGED, final layout")
print("="*65)

s1 = np.load('results/multimethod_stage1.npz', allow_pickle=True)
s2_oi = np.load('results/multimethod_oi.npz', allow_pickle=True)
s2_dineof = np.load('results/multimethod_dineof.npz', allow_pickle=True)
s3_mc = np.load('results/multimethod_montecarlo.npz', allow_pickle=True)

chl_norm = np.load('data/modis_chl/chl_norm.npy')
chl_mean = np.load('data/modis_chl/chl_mean.npy')[0]
chl_std = np.load('data/modis_chl/chl_std.npy')[0]
l4_norm = np.load('data/l4_chl/chl_l4_norm.npy')
dates = np.load('data/modis_chl/chl_dates.npy')
lat = np.load('data/modis_chl/chl_lat.npy')
lon = np.load('data/modis_chl/chl_lon.npy')

print(f"MODIS chl_norm shape: {chl_norm.shape}")
print(f"L4 chl_l4_norm shape: {l4_norm.shape}")
assert chl_norm.shape == l4_norm.shape, "MODIS/L4 shape mismatch -- do not proceed blindly"

target_dates = list(s1['target_dates'])
target_idx = list(s1['target_idx'])
land_mask = s1['land_mask']
ocean_mask = s1['ocean_mask']
extent = [lon.min(), lon.max(), lat.min(), lat.max()]

# Human-readable date labels, e.g. "August 29, 2019" -- plain string
# formatting, not a platform-specific strftime flag, so this can't
# silently behave differently across machines.
date_labels = {}
for d in target_dates:
    dt = datetime.strptime(str(d), '%Y%m%d')
    date_labels[d] = f"{dt.strftime('%B')} {dt.day}, {dt.year}"
    print(f"  {d} -> {date_labels[d]}")

for d, idx in zip(target_dates, target_idx):
    n_nan_l4 = int(np.isnan(l4_norm[idx]).sum())
    print(f"  {d}: L4 NaN pixel count = {n_nan_l4} (should be near 0 for a filled row)")

coverage = {}
for d, idx in zip(target_dates, target_idx):
    valid_today = ~np.isnan(chl_norm[idx])
    coverage[d] = 100 * (ocean_mask & valid_today).sum() / ocean_mask.sum()

cmap_gray = ListedColormap(['#8c8c8c'])
cmap_white = ListedColormap(['#ffffff'])
cmap_data = plt.get_cmap('RdBu_r')

def plot_3layer(ax, field, land_mask, extent):
    ones = np.ones(field.shape)
    ax.imshow(ones, extent=extent, origin='upper', cmap=cmap_gray,
               vmin=0, vmax=1, aspect='equal')
    ocean_white = np.ma.masked_array(ones, mask=land_mask)
    ax.imshow(ocean_white, extent=extent, origin='upper', cmap=cmap_white,
               vmin=0, vmax=1, aspect='equal')
    valid_today = ~np.isnan(field)
    colored = np.ma.masked_array(field, mask=~valid_today)
    im = ax.imshow(colored, extent=extent, origin='upper', cmap=cmap_data,
                    vmin=-3, vmax=3, aspect='equal')
    return im

def plot_2layer(ax, field, land_mask, extent):
    ones = np.ones(field.shape)
    ax.imshow(ones, extent=extent, origin='upper', cmap=cmap_gray,
               vmin=0, vmax=1, aspect='equal')
    colored = np.ma.masked_array(field, mask=land_mask)
    im = ax.imshow(colored, extent=extent, origin='upper', cmap=cmap_data,
                    vmin=-3, vmax=3, aspect='equal')
    return im

COL_LETTERS = ['a', 'b', 'c', 'd']

def build_figure(row_specs, out_path, title):
    n_rows, n_cols = len(row_specs), len(target_dates)
    data_h = lat.max() - lat.min()
    data_w = lon.max() - lon.min()
    panel_w = 3.0
    panel_h = panel_w * (data_h / data_w)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_w*n_cols, panel_h*n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    last_im = None
    for col, d in enumerate(target_dates):
        for row, (label, kind, data_dict) in enumerate(row_specs):
            ax = axes[row][col]
            field = data_dict[d]
            if kind == '3layer':
                im = plot_3layer(ax, field, land_mask, extent)
            else:
                im = plot_2layer(ax, field, land_mask, extent)
            last_im = im

            if row == 0:
                panel_title = (f'({COL_LETTERS[col]}) {date_labels[d]}\n'
                                f'{coverage[d]:.1f}% coverage')
                ax.set_title(panel_title, fontsize=9, fontweight='bold')

            if col == 0:
                ax.set_ylabel(label.replace('\n', ' '), fontsize=9, fontweight='bold')
                ax.set_yticks(np.linspace(lat.min(), lat.max(), 4).round(0))
                ax.tick_params(axis='y', labelsize=7)
            else:
                ax.set_yticks([])
            if row == n_rows - 1:
                ax.set_xticks(np.linspace(lon.min(), lon.max(), 4).round(0))
                ax.tick_params(axis='x', labelsize=7)
            else:
                ax.set_xticks([])

    fig.text(0.06, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=11)
    fig.text(0.47, 0.01, 'Longitude', ha='center', fontsize=11)

    plt.subplots_adjust(wspace=0.05, hspace=0.28, top=0.95, bottom=0.04,
                         left=0.09, right=0.90)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.998)

    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(last_im, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Normalized log$_{10}$(Chl-a)', fontsize=9)

    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f"Saved -> {out_path}")

mask_data = {d: chl_norm[idx] for d, idx in zip(target_dates, target_idx)}
oi_data = {d: s2_oi[f'oi_{d}'] for d in target_dates}
dineof_data = {d: s2_dineof[f'dineof_{d}'] for d in target_dates}
cellnn_data = {d: s1[f'cellnn_{d}'] for d in target_dates}
mc_mean_data = {d: (s3_mc[f'mcmean_{d}'] - chl_mean) / chl_std for d in target_dates}
mc_draw1_data = {d: (s3_mc[f'mcdraw1_{d}'] - chl_mean) / chl_std for d in target_dates}
mc_draw2_data = {d: (s3_mc[f'mcdraw2_{d}'] - chl_mean) / chl_std for d in target_dates}
gt_l4_data = {d: l4_norm[idx] for d, idx in zip(target_dates, target_idx)}

merged_rows = [
    ('Mask (MODIS L3)', '3layer', mask_data),
    ('Filled ground truth (L4)', '2layer', gt_l4_data),
    ('OI', '2layer', oi_data),
    ('DINEOF', '2layer', dineof_data),
    ('Cell-NN + Diffusion', '2layer', cellnn_data),
    ('Monte Carlo (mean)', '2layer', mc_mean_data),
    ('Monte Carlo (draw 1)', '2layer', mc_draw1_data),
    ('Monte Carlo (draw 2)', '2layer', mc_draw2_data),
]
build_figure(merged_rows, 'results/multimethod_merged.png',
             'Gap-filling method comparison, Bay of Bengal')

print("\nDone.")
