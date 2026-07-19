"""
Multi-method comparison figure -- v2 rebuild
================================================
Fixes from v1:
  - Mask/ground-truth rows now show REAL colored Chl-a data (3-layer
    compositing: gray land, white cloud gap, colored valid data),
    matching Mam's convention -- not an abstract categorical scheme.
  - Reconstruction rows use 2-layer compositing (gray land, colored
    everywhere else -- no white gaps since they're fully filled).
  - True geographic aspect ratio (no stretching).
  - Fixed row-label overlap.
  - Split into two figures: deterministic methods, and Monte Carlo
    stochastic samples.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

print("="*65)
print("Multi-method figure v2 -- deterministic + stochastic")
print("="*65)

s1 = np.load('results/multimethod_stage1.npz', allow_pickle=True)
s2_oi = np.load('results/multimethod_oi.npz', allow_pickle=True)
s2_dineof = np.load('results/multimethod_dineof.npz', allow_pickle=True)
s3_mc = np.load('results/multimethod_montecarlo.npz', allow_pickle=True)

chl_norm = np.load('data/modis_chl/chl_norm.npy')
chl_mean = np.load('data/modis_chl/chl_mean.npy')[0]
chl_std = np.load('data/modis_chl/chl_std.npy')[0]
dates = np.load('data/modis_chl/chl_dates.npy')
lat = np.load('data/modis_chl/chl_lat.npy')
lon = np.load('data/modis_chl/chl_lon.npy')

target_dates = list(s1['target_dates'])
target_idx = list(s1['target_idx'])
land_mask = s1['land_mask']
ocean_mask = s1['ocean_mask']
extent = [lon.min(), lon.max(), lat.min(), lat.max()]

coverage = {}
for d, idx in zip(target_dates, target_idx):
    valid_today = ~np.isnan(chl_norm[idx])
    coverage[d] = 100 * (ocean_mask & valid_today).sum() / ocean_mask.sum()

cmap_gray = ListedColormap(['#8c8c8c'])
cmap_white = ListedColormap(['#ffffff'])
cmap_data = plt.get_cmap('RdBu_r')

def plot_3layer(ax, field, land_mask, extent):
    """Raw/gappy fields: gray land, white cloud-gap, colored valid data."""
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
    """Reconstructions: gray land, colored everywhere else."""
    ones = np.ones(field.shape)
    ax.imshow(ones, extent=extent, origin='upper', cmap=cmap_gray,
               vmin=0, vmax=1, aspect='equal')
    colored = np.ma.masked_array(field, mask=land_mask)
    im = ax.imshow(colored, extent=extent, origin='upper', cmap=cmap_data,
                    vmin=-3, vmax=3, aspect='equal')
    return im

def build_figure(row_specs, out_path, title):
    """row_specs: list of (label, kind, data_dict) where kind is '3layer' or '2layer'"""
    n_rows, n_cols = len(row_specs), len(target_dates)
    # Data aspect ratio (height/width in degrees) drives figsize directly,
    # so subplot boxes match the actual rendered content -- no dead space.
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

            # Per-panel title: row 0 combines date+coverage with the row
            # label; every other row repeats its own method name -- both
            # match Mam's convention of labeling every panel, not just
            # the outer edges.
            if row == 0:
                panel_title = f'{d}, {coverage[d]:.1f}% coverage\n{label}'
            else:
                panel_title = label.replace('\n', ' ')
            ax.set_title(panel_title, fontsize=9, fontweight='bold')

            # Axis ticks: real lat/lon reference, shown only on the outer
            # edges (left column gets latitude, bottom row gets
            # longitude) to avoid cluttering every panel.
            if col == 0:
                ax.set_yticks(np.linspace(lat.min(), lat.max(), 4).round(0))
                ax.tick_params(axis='y', labelsize=7)
            else:
                ax.set_yticks([])
            if row == n_rows - 1:
                ax.set_xticks(np.linspace(lon.min(), lon.max(), 4).round(0))
                ax.tick_params(axis='x', labelsize=7)
            else:
                ax.set_xticks([])
            if col == 0:
                ax.set_ylabel('Latitude', fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel('Longitude', fontsize=8)

    plt.subplots_adjust(wspace=0.05, hspace=0.25, top=0.90, bottom=0.08)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)

    # Shared colorbar: one range (-3 to 3) applies to every reconstruction
    # panel, so a single shared colorbar is correct here, not per-column.
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    cbar = fig.colorbar(last_im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Normalized log$_{10}$(Chl-a)', fontsize=9)

    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f"Saved -> {out_path}")

# ── Build data dictionaries ──
mask_data = {d: chl_norm[idx] for d, idx in zip(target_dates, target_idx)}
bg_data = {d: s1[f'bg_{d}'] for d in target_dates}
oi_data = {d: s2_oi[f'oi_{d}'] for d in target_dates}
dineof_data = {d: s2_dineof[f'dineof_{d}'] for d in target_dates}
cellnn_data = {d: s1[f'cellnn_{d}'] for d in target_dates}
gt_data = {d: chl_norm[idx] for d, idx in zip(target_dates, target_idx)}

mc_mean_data = {d: (s3_mc[f'mcmean_{d}'] - chl_mean) / chl_std for d in target_dates}
mc_draw1_data = {d: (s3_mc[f'mcdraw1_{d}'] - chl_mean) / chl_std for d in target_dates}
mc_draw2_data = {d: (s3_mc[f'mcdraw2_{d}'] - chl_mean) / chl_std for d in target_dates}

# ── Figure A: deterministic methods ──
deterministic_rows = [
    ('Mask', '3layer', mask_data),
    ('OI', '2layer', oi_data),
    ('DINEOF', '2layer', dineof_data),
    ('Cell-NN\n+ Diffusion', '2layer', cellnn_data),
    ('Gappy\nground truth', '3layer', gt_data),
]
build_figure(deterministic_rows, 'results/multimethod_deterministic.png',
             'Deterministic gap-filling comparison, Bay of Bengal')

# ── Figure B: stochastic Monte Carlo ──
stochastic_rows = [
    ('Mask', '3layer', mask_data),
    ('Monte Carlo\n(mean)', '2layer', mc_mean_data),
    ('Monte Carlo\n(draw 1)', '2layer', mc_draw1_data),
    ('Monte Carlo\n(draw 2)', '2layer', mc_draw2_data),
]
build_figure(stochastic_rows, 'results/multimethod_stochastic.png',
             'Monte Carlo stochastic sampling, Bay of Bengal')

print("\nBoth figures done.")
