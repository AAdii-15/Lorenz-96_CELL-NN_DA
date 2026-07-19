"""
Multi-method comparison figure -- Stage 1
=============================================
Builds the mask panel, Background, and Cell-NN+Diffusion rows for the
4 selected dates. Saves intermediate results so later stages (OI,
DINEOF, Monte Carlo) can be added without recomputing this stage.
"""
import numpy as np
import matplotlib.pyplot as plt

print("="*65)
print("Multi-method figure -- Stage 1: mask, Background, Cell-NN")
print("="*65)

chl_norm = np.load('data/modis_chl/chl_norm.npy')
chl_reconstructed = np.load('data/modis_chl/chl_reconstructed.npy')
dates = np.load('data/modis_chl/chl_dates.npy')
lat = np.load('data/modis_chl/chl_lat.npy')
lon = np.load('data/modis_chl/chl_lon.npy')

date_strs = [str(d) for d in dates]
target_dates = ['20190829', '20190914', '20170501', '20170125']
target_idx = [date_strs.index(d) for d in target_dates]
print(f"Target dates and indices: {list(zip(target_dates, target_idx))}")

ocean_mask = np.any(~np.isnan(chl_norm), axis=0)
land_mask = ~ocean_mask

# Mask panel: 0=land, 1=cloud gap (ocean, missing today), 2=input (ocean, valid today)
masks = {}
backgrounds = {}
cellnn_outputs = {}
coverages = {}

for d, idx in zip(target_dates, target_idx):
    field_curr = chl_norm[idx]
    field_prev = chl_norm[idx - 1]
    valid_today = ~np.isnan(field_curr)

    category = np.zeros(land_mask.shape, dtype=int)
    category[land_mask] = 0
    category[ocean_mask & ~valid_today] = 1
    category[ocean_mask & valid_today] = 2
    masks[d] = category

    backgrounds[d] = field_prev
    cellnn_outputs[d] = chl_reconstructed[idx]

    cov = 100 * (ocean_mask & valid_today).sum() / ocean_mask.sum()
    coverages[d] = cov
    print(f"  {d}: ocean coverage today = {cov:.1f}%")

np.savez('results/multimethod_stage1.npz',
         target_dates=target_dates, target_idx=target_idx,
         **{f'mask_{d}': masks[d] for d in target_dates},
         **{f'bg_{d}': backgrounds[d] for d in target_dates},
         **{f'cellnn_{d}': cellnn_outputs[d] for d in target_dates},
         land_mask=land_mask, ocean_mask=ocean_mask)
print("\nSaved -> results/multimethod_stage1.npz")

# Quick preview: mask row + Cell-NN row for all 4 dates
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
from matplotlib.colors import ListedColormap, BoundaryNorm
cmap_mask = ListedColormap(['#8c8c8c', '#ffffff', '#4daf4a'])
norm_mask = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap_mask.N)

for col, d in enumerate(target_dates):
    axes[0][col].imshow(masks[d], extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                         origin='upper', cmap=cmap_mask, norm=norm_mask, aspect='auto')
    axes[0][col].set_title(f'{d}\n{coverages[d]:.1f}% coverage', fontsize=10)
    axes[1][col].imshow(cellnn_outputs[d], extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                         origin='upper', cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')

axes[0][0].set_ylabel('Mask', fontsize=11, fontweight='bold')
axes[1][0].set_ylabel('Cell-NN', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('results/multimethod_stage1_preview.png', dpi=120, bbox_inches='tight')
plt.show()
print("Saved -> results/multimethod_stage1_preview.png")
