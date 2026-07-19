"""
Multi-method comparison figure -- Final assembly
====================================================
Combines Stages 1-3 into the full 9-row x 4-column comparison figure,
matching Mam's earlier paper's convention.

Rows: Mask, Background, OI, DINEOF, Cell-NN+Diffusion, Monte Carlo
      (mean), Monte Carlo (draw 1), Monte Carlo (draw 2), Gappy ground
      truth.
Columns: 4 dates spanning low-to-high cloud coverage.

Land is masked out (shown gray, matching the mask row's convention)
in every reconstruction row, since none of these methods distinguish
land from ocean on their own -- only ocean_mask does.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

print("="*65)
print("Multi-method figure -- Final assembly")
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
date_strs = [str(d) for d in dates]

target_dates = list(s1['target_dates'])
target_idx = list(s1['target_idx'])
land_mask = s1['land_mask']
ocean_mask = s1['ocean_mask']

coverage = {}
for d, idx in zip(target_dates, target_idx):
    valid_today = ~np.isnan(chl_norm[idx])
    coverage[d] = 100 * (ocean_mask & valid_today).sum() / ocean_mask.sum()

def land_out(field):
    f = field.copy()
    f[land_mask] = np.nan
    return f

# ── Assemble all rows, per date, with land masked and MC normalized ──
row_data = {}
for d, idx in zip(target_dates, target_idx):
    mc_mean_norm = (s3_mc[f'mcmean_{d}'] - chl_mean) / chl_std
    mc_draw1_norm = (s3_mc[f'mcdraw1_{d}'] - chl_mean) / chl_std
    mc_draw2_norm = (s3_mc[f'mcdraw2_{d}'] - chl_mean) / chl_std

    row_data[d] = {
        'mask': s1[f'mask_{d}'],
        'background': land_out(s1[f'bg_{d}']),
        'oi': land_out(s2_oi[f'oi_{d}']),
        'dineof': land_out(s2_dineof[f'dineof_{d}']),
        'cellnn': land_out(s1[f'cellnn_{d}']),
        'mc_mean': land_out(mc_mean_norm),
        'mc_draw1': land_out(mc_draw1_norm),
        'mc_draw2': land_out(mc_draw2_norm),
        'ground_truth': land_out(chl_norm[idx]),
    }

# ── Build the figure ──
row_labels = ['Mask', 'Background', 'OI', 'DINEOF', 'Cell-NN\n+ Diffusion',
              'Monte Carlo\n(mean)', 'Monte Carlo\n(draw 1)',
              'Monte Carlo\n(draw 2)', 'Gappy\nground truth']
row_keys = ['mask', 'background', 'oi', 'dineof', 'cellnn',
            'mc_mean', 'mc_draw1', 'mc_draw2', 'ground_truth']

n_rows, n_cols = len(row_keys), len(target_dates)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))

cmap_mask = ListedColormap(['#8c8c8c', '#ffffff', '#4daf4a'])
norm_mask = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_mask.N)

# RdBu_r with explicit gray for masked (land) pixels -- NaN alone renders
# as white by default in matplotlib, which is indistinguishable from
# "missing data" white. Using a masked array + set_bad forces land to
# render as the same gray used in the mask row, consistently.
import copy
cmap_data = copy.copy(plt.get_cmap('RdBu_r'))
cmap_data.set_bad('#8c8c8c')

for col, d in enumerate(target_dates):
    axes[0][col].set_title(f'{d}\n{coverage[d]:.1f}% coverage', fontsize=11, fontweight='bold')
    for row, (label, key) in enumerate(zip(row_labels, row_keys)):
        ax = axes[row][col]
        if key == 'mask':
            im = ax.imshow(row_data[d][key], extent=[lon.min(),lon.max(),lat.min(),lat.max()],
                            origin='upper', cmap=cmap_mask, norm=norm_mask, aspect='auto')
        else:
            masked = np.ma.masked_invalid(row_data[d][key])
            im = ax.imshow(masked, extent=[lon.min(),lon.max(),lat.min(),lat.max()],
                            origin='upper', cmap=cmap_data, vmin=-3, vmax=3, aspect='auto')
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(label, fontsize=10, fontweight='bold', rotation=90, labelpad=8)

plt.tight_layout()
plt.savefig('results/multimethod_comparison.png', dpi=130, bbox_inches='tight')
plt.show()
print("\nSaved -> results/multimethod_comparison.png")
