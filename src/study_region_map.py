"""
Study Region Map -- Bay of Bengal
=====================================
Matches the convention from Mam's earlier paper (Fig. 1):
  Gray  = land (never a valid ocean pixel across the whole record)
  Green = region with satellite data (valid on the shown date)
  White = missing data area (ocean, but cloud-covered on the shown date)

No red evaluation box yet -- Task 1's region is still pending review.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

print("="*65)
print("Study Region Map -- Bay of Bengal")
print("="*65)

chl_norm = np.load('data/modis_chl/chl_norm.npy')
dates    = np.load('data/modis_chl/chl_dates.npy')
lat      = np.load('data/modis_chl/chl_lat.npy')
lon      = np.load('data/modis_chl/chl_lon.npy')

target_date = '20150101'
date_strs = [str(d) for d in dates]
idx = date_strs.index(target_date)
print(f"Target date: {dates[idx]} (composite index {idx})")

# Land vs ocean: ocean = valid at least once across the full 230-composite record
ocean_mask = np.any(~np.isnan(chl_norm), axis=0)
land_mask  = ~ocean_mask

# Valid today vs missing today (only meaningful for ocean pixels)
valid_today   = ~np.isnan(chl_norm[idx])
missing_today = ocean_mask & ~valid_today

n_land    = land_mask.sum()
n_valid   = (ocean_mask & valid_today).sum()
n_missing = missing_today.sum()
n_ocean   = ocean_mask.sum()
print(f"Land pixels: {n_land}")
print(f"Ocean pixels: {n_ocean}")
print(f"  Valid today: {n_valid} ({100*n_valid/n_ocean:.1f}% of ocean)")
print(f"  Missing today (cloud gap): {n_missing} ({100*n_missing/n_ocean:.1f}% of ocean)")

# Category map: 0=land, 1=valid data, 2=missing
category_map = np.zeros(land_mask.shape, dtype=int)
category_map[land_mask] = 0
category_map[ocean_mask & valid_today] = 1
category_map[missing_today] = 2

cmap = ListedColormap(['#8c8c8c', '#4daf4a', '#ffffff'])  # gray, green, white
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = BoundaryNorm(bounds, cmap.N)

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(category_map, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
               origin='upper', cmap=cmap, norm=norm, aspect='auto')
ax.set_xlabel('Longitude', fontsize=11)
ax.set_ylabel('Latitude', fontsize=11)
ax.set_title(f'Bay of Bengal study region -- {dates[idx][:4]}-{dates[idx][4:6]}-{dates[idx][6:]}',
             fontsize=12, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.8)
cbar.ax.set_yticklabels(['Land', 'Data', 'Missing'], fontsize=10)

plt.tight_layout()
plt.savefig('results/study_region_map.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved -> results/study_region_map.png")
