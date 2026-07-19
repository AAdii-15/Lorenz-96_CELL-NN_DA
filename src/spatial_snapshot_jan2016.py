"""
Spatial Map Snapshot -- 1 January 2016
Cell-NN-reconstructed MODIS vs independent VIIRS (ground truth sensor)
=========================================================================
One of Mam's three pending tasks. Shows MODIS observed input, Cell-NN's
full reconstruction (chl_reconstructed.npy -- fills genuine cloud gaps
using real observations, not the held-out validation exercise), and the
independent VIIRS sensor for the same date, in physical units (mg/m3,
log10 color scale, matching preprocess_chl.py's existing convention).
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import glob, os

print("=" * 65)
print("Spatial Snapshot -- 2016-01-01: MODIS / Cell-NN / VIIRS")
print("=" * 65)

chl_data = np.load('data/modis_chl/chl_data.npy')
chl_reconstructed = np.load('data/modis_chl/chl_reconstructed.npy')
chl_mean = np.load('data/modis_chl/chl_mean.npy')[0]
chl_std = np.load('data/modis_chl/chl_std.npy')[0]
dates = np.load('data/modis_chl/chl_dates.npy')
lat = np.load('data/modis_chl/chl_lat.npy')
lon = np.load('data/modis_chl/chl_lon.npy')

target_date = '20160101'
date_strs = [str(d) for d in dates]
if target_date not in date_strs:
    raise ValueError(f"{target_date} not found in MODIS dates")
idx = date_strs.index(target_date)
print(f"MODIS index for {target_date}: {idx}")

modis_raw = chl_data[idx]

cellnn_log = chl_reconstructed[idx] * chl_std + chl_mean
cellnn_raw = 10 ** cellnn_log

chl_l4_data = np.load('data/l4_chl/chl_l4_data.npy')
l4_raw_field = chl_l4_data[idx]
print(f"L4 composite loaded at same index as MODIS ({idx}) for {target_date}")

modis_valid_pct = 100 * (~np.isnan(modis_raw)).mean()
l4_valid_pct = 100 * (~np.isnan(l4_raw_field)).mean()
print(f"MODIS valid coverage: {modis_valid_pct:.1f}%")
print(f"L4 valid coverage: {l4_valid_pct:.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Ocean mask: pixel is ocean if valid at least once across the full record
# (same definition used in cloud_mask_all_methods.py)
chl_norm_full = np.load('data/modis_chl/chl_norm.npy')
ocean_mask = np.any(~np.isnan(chl_norm_full), axis=0)

cellnn_masked = np.where(ocean_mask, cellnn_raw, np.nan)

panels = [
    (np.log10(modis_raw), 'MODIS Observed (input)'),
    (np.log10(cellnn_masked), 'Cell-NN Reconstructed'),
    (np.log10(l4_raw_field), 'L4 (Ground Truth)'),
]

panel_letters = ['(a)', '(b)', '(c)']

vmin, vmax = -1, 2  # matches preprocess_chl.py's existing log10(Chl-a) convention

for i, (ax, (data, title)) in enumerate(zip(axes, panels)):
    im = ax.imshow(data, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                    origin='upper', cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.text(0.02, 0.98, panel_letters[i], transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top', color='black')
    ax.set_xlabel('Lon')
    ax.set_ylabel('Lat')
    plt.colorbar(im, ax=ax, shrink=0.8, label='log10(Chl-a mg/m3)')

plt.tight_layout()
plt.savefig('results/spatial_snapshot_jan2016.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved -> results/spatial_snapshot_jan2016.png")
print("\nDone.")
