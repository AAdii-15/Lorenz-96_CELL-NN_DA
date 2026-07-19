"""
Task 2: Visualize MODIS Chlorophyll-a Spatial Maps
Bay of Bengal: 80-100E, 5-22N
Period: 2015-2019, 8-day
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
import os
import glob
from datetime import datetime

print("="*60)
print("Task 2: MODIS Chl-a Visualization")
print("="*60)

DATA_DIR = "data/modis_chl/nc_files"
files    = sorted(glob.glob(f"{DATA_DIR}/*.nc"))
print(f"Total files: {len(files)}")

# ── Load all files into array ─────────────────────────
print("\nLoading all 230 files...")
chl_list = []
dates    = []

for f in files:
    ds  = xr.open_dataset(f)
    chl = ds['chlor_a'].values  # shape (408, 480)
    chl_list.append(chl)

    # Extract date from filename
    fname = os.path.basename(f)
    date_str = fname.split('.')[1].split('_')[0]  # e.g. 20150101
    dates.append(datetime.strptime(date_str, '%Y%m%d'))
    ds.close()

# Stack into 3D array: (time, lat, lon)
chl_data = np.array(chl_list)  # shape (230, 408, 480)
print(f"Data shape: {chl_data.shape}")
print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

# Get lat/lon from last file
ds  = xr.open_dataset(files[0])
lat = ds['lat'].values
lon = ds['lon'].values
ds.close()

# ── Basic Statistics ──────────────────────────────────
print(f"\nData Statistics:")
print(f"  Min  : {np.nanmin(chl_data):.4f} mg/m³")
print(f"  Max  : {np.nanmax(chl_data):.4f} mg/m³")
print(f"  Mean : {np.nanmean(chl_data):.4f} mg/m³")
print(f"  NaN% : {100*np.isnan(chl_data).mean():.1f}%")

# ── PLOT 1: 12 Spatial Maps (one per 5 months) ────────
print("\nPlot 1: Spatial maps...")
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

# Pick 12 evenly spaced time steps
indices = np.linspace(0, 229, 12, dtype=int)

for idx, ax in zip(indices, axes):
    chl = chl_data[idx]
    # Log transform for visualization
    chl_log = np.log10(chl)
    chl_log[~np.isfinite(chl_log)] = np.nan

    im = ax.imshow(chl_log,
                   extent=[lon.min(), lon.max(),
                           lat.min(), lat.max()],
                   origin='upper',
                   cmap='jet',
                   vmin=-1, vmax=2,
                   aspect='auto')
    ax.set_title(dates[idx].strftime('%Y-%m-%d'),
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Lon', fontsize=8)
    ax.set_ylabel('Lat', fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8,
                 label='log₁₀(Chl-a)')

fig.suptitle('MODIS Aqua Chlorophyll-a — Bay of Bengal\n'
             '8-day Composite (2015-2019) | log₁₀ scale',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/chl_spatial_maps.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved → results/chl_spatial_maps.png")

# ── PLOT 2: Time series — mean Chl-a over region ──────
print("\nPlot 2: Time series...")
mean_chl = np.nanmean(chl_data, axis=(1,2))  # (230,)

fig, axes = plt.subplots(2, 1, figsize=(16, 8))

# Raw
axes[0].plot(dates, mean_chl, 'b-', lw=1.0)
axes[0].set_title('Mean Chl-a over Bay of Bengal (raw)',
                  fontsize=12)
axes[0].set_ylabel('Chl-a (mg/m³)')
axes[0].grid(True, alpha=0.3)

# Log scale
axes[1].plot(dates, np.log10(mean_chl), 'g-', lw=1.0)
axes[1].set_title('Mean Chl-a over Bay of Bengal (log₁₀)',
                  fontsize=12)
axes[1].set_ylabel('log₁₀(Chl-a)')
axes[1].set_xlabel('Date')
axes[1].grid(True, alpha=0.3)

# Shade years
for year in [2015, 2016, 2017, 2018, 2019]:
    if year % 2 == 0:
        axes[0].axvspan(datetime(year,1,1),
                        datetime(year,12,31),
                        alpha=0.1, color='gray')
        axes[1].axvspan(datetime(year,1,1),
                        datetime(year,12,31),
                        alpha=0.1, color='gray')

plt.tight_layout()
plt.savefig('results/chl_timeseries.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved → results/chl_timeseries.png")

# ── PLOT 3: Seasonal mean maps ─────────────────────────
print("\nPlot 3: Seasonal maps...")
seasons = {
    'Winter (DJF)': [12, 1, 2],
    'Spring (MAM)': [3, 4, 5],
    'Summer (JJA)': [6, 7, 8],
    'Autumn (SON)': [9, 10, 11]
}

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, (season_name, months) in zip(axes, seasons.items()):
    # Find indices for this season
    mask = [d.month in months for d in dates]
    chl_season = np.nanmean(chl_data[mask], axis=0)
    chl_log    = np.log10(chl_season)
    chl_log[~np.isfinite(chl_log)] = np.nan

    im = ax.imshow(chl_log,
                   extent=[lon.min(), lon.max(),
                           lat.min(), lat.max()],
                   origin='upper',
                   cmap='jet',
                   vmin=-1, vmax=2,
                   aspect='auto')
    ax.set_title(season_name, fontsize=12,
                 fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.colorbar(im, ax=ax, shrink=0.8,
                 label='log₁₀(Chl-a)')

fig.suptitle('Seasonal Mean Chl-a — Bay of Bengal (2015-2019)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/chl_seasonal_maps.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved → results/chl_seasonal_maps.png")

# ── Save processed data ───────────────────────────────
print("\nSaving processed data...")
np.save('data/modis_chl/chl_data.npy',  chl_data)
np.save('data/modis_chl/chl_lat.npy',   lat)
np.save('data/modis_chl/chl_lon.npy',   lon)
np.save('data/modis_chl/chl_dates.npy',
        np.array([d.strftime('%Y%m%d') for d in dates]))

print("Saved → data/modis_chl/chl_data.npy")
print(f"Shape: {chl_data.shape} = (time, lat, lon)")
print("\nTask 2 COMPLETE!")
