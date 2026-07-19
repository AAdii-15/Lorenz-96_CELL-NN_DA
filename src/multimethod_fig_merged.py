"""
Multi-method comparison figure -- MERGED, v2 layout
=======================================================
Redesign per feedback:
  1. Latitude/Longitude axis LABEL TEXT shown once for the whole
     figure (numeric tick values on the outer edges are unchanged).
  2. Colorbar moved to a single vertical bar on the right.
  3. Column headers (a)-(d) label the four dates once, at the top;
     row labels (method names) shown once per row, to the left of
     the leftmost panel, instead of repeated in every panel's title.
  4. Filled ground truth (L4) row moved to position 2 (right after
     Mask), not last.
  5. Mask row explicitly notes it is derived from MODIS L3 coverage.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import string

print("="*65)
print("Multi-method figure -- MERGED v2 layout")
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

def build_figure(row_specs, out_path, title):
    n_rows, n_cols = len(row_specs), len(target_dates)
    data_h = lat.max() - lat.min()
    data_w = lon.max() - lon.min()
    panel_w = 3.0
    panel_h = panel_w * (data_h / data_w)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_w*n_cols + 0.9, panel_h*n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_letters = string.ascii_lowercase[:n_cols]
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

            # Column headers: only on row 0, one per column -- (a)/(b)/(c)/(d)
            # plus date and coverage. Method names are row labels now,
            # shown once per row instead of repeated per panel.
            if row == 0:
                ax.set_title(f'({col_letters[col]}) {d}\n{coverage[d]:.1f}% coverage',
                             fontsize=9, fontweight='bold')

            # Row labels: once per row, placed left of the leftmost panel
            # using axes-relative coordinates, robust to final layout.
            if col == 0:
                ax.text(-0.38, 0.5, label, transform=ax.transAxes,
                        rotation=90, va='center', ha='center',
                        fontsize=9, fontweight='bold')

            # Numeric tick VALUES kept on outer edges, same as before --
            # only the repeated "Latitude"/"Longitude" word labels change.
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

    # Single shared axis labels for the whole figure, not per row/column.
    fig.text(0.47, 0.01, 'Longitude', ha='center', fontsize=11)
    fig.text(0.015, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=11)

    plt.subplots_adjust(left=0.10, right=0.87, wspace=0.05, hspace=0.35, top=0.95, bottom=0.04)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.998)

    # Single vertical colorbar on the right.
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(last_im, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Normalized log$_{10}$(Chl-a)', fontsize=9)

    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f"Saved -> {out_path}")

# ── Build data dictionaries ──
mask_data = {d: chl_norm[idx] for d, idx in zip(target_dates, target_idx)}
oi_data = {d: s2_oi[f'oi_{d}'] for d in target_dates}
dineof_data = {d: s2_dineof[f'dineof_{d}'] for d in target_dates}
cellnn_data = {d: s1[f'cellnn_{d}'] for d in target_dates}
mc_mean_data = {d: (s3_mc[f'mcmean_{d}'] - chl_mean) / chl_std for d in target_dates}
mc_draw1_data = {d: (s3_mc[f'mcdraw1_{d}'] - chl_mean) / chl_std for d in target_dates}
mc_draw2_data = {d: (s3_mc[f'mcdraw2_{d}'] - chl_mean) / chl_std for d in target_dates}
gt_l4_data = {d: l4_norm[idx] for d, idx in zip(target_dates, target_idx)}

# Ground truth now row 2 (right after Mask); Mask row notes MODIS L3 source.
merged_rows = [
    ('Mask\n(MODIS L3)', '3layer', mask_data),
    ('Filled\nground truth (L4)', '2layer', gt_l4_data),
    ('OI', '2layer', oi_data),
    ('DINEOF', '2layer', dineof_data),
    ('Cell-NN\n+ Diffusion', '2layer', cellnn_data),
    ('Monte Carlo\n(mean)', '2layer', mc_mean_data),
    ('Monte Carlo\n(draw 1)', '2layer', mc_draw1_data),
    ('Monte Carlo\n(draw 2)', '2layer', mc_draw2_data),
]
print("\nRow order:", [r[0].replace(chr(10), ' ') for r in merged_rows])

build_figure(merged_rows, 'results/multimethod_merged.png',
             'Gap-filling method comparison, Bay of Bengal')

print("\nDone.")
