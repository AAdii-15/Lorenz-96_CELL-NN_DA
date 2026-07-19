"""
Preprocess L4 gap-free Chl-a (Copernicus Marine) to match MODIS pipeline.
1. Aggregate daily L4 files into 8-day composites (aligned to MODIS dates)
2. Log10 transformation
3. Z-score normalization (independent from MODIS/VIIRS statistics)
4. Save preprocessed data
"""
import numpy as np
import xarray as xr
import os
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore', message='Mean of empty slice')

print("="*60)
print("Preprocessing L4 Chl-a (Copernicus gap-free)")
print("="*60)

L4_DIR = '/Volumes/TOSHIBA EXT/data_aditya'
MODIS_DATES_PATH = 'data/modis_chl/chl_dates.npy'

# ── Step 1: Load MODIS composite dates (defines our windows) ──
dates_str = np.load(MODIS_DATES_PATH, allow_pickle=True)
modis_dates = [datetime.strptime(str(d), '%Y%m%d') for d in dates_str]
print(f"MODIS composite dates loaded: {len(modis_dates)}")

def load_l4_day(date):
    fpath = os.path.join(L4_DIR, date.strftime('%Y_%m_%d.nc'))
    if not os.path.exists(fpath):
        return None
    ds = xr.open_dataset(fpath)
    chl = ds['CHL'].values[0]      # (408, 480)
    chl = chl[::-1, :]              # flip latitude to match MODIS N->S orientation
    ds.close()
    return chl

# ── Step 2: Build 8-day composites for each MODIS window ──
print("\nBuilding 8-day composites...")
l4_composites = []
composite_coverage = []

for i, start in enumerate(modis_dates):
    if i + 1 < len(modis_dates):
        end = modis_dates[i + 1] - timedelta(days=1)
    else:
        end = start + timedelta(days=7)

    window_dates = [start + timedelta(days=k) for k in range((end - start).days + 1)]
    daily_fields = [load_l4_day(d) for d in window_dates]
    n_found = sum(f is not None for f in daily_fields)
    daily_fields = [f for f in daily_fields if f is not None]

    if len(daily_fields) == 0:
        composite = np.full((408, 480), np.nan)
    else:
        composite = np.nanmean(np.stack(daily_fields), axis=0)

    l4_composites.append(composite)
    composite_coverage.append(n_found / len(window_dates))

    if n_found < len(window_dates):
        print(f"  {start.date()}: only {n_found}/{len(window_dates)} daily files found")

l4_data = np.stack(l4_composites)  # (230, 408, 480)
print(f"\nL4 composite stack shape: {l4_data.shape}")
print(f"Mean daily-file coverage per composite: {np.mean(composite_coverage)*100:.1f}%")

# ── Step 3: Log10 transformation (identical to preprocess_chl.py) ──
print("\nStep: Log10 transformation...")
l4_log = np.log10(l4_data)
l4_log[~np.isfinite(l4_log)] = np.nan
print(f"Log range: {np.nanmin(l4_log):.3f} to {np.nanmax(l4_log):.3f}")

# ── Step 4: Z-score normalization (independent statistics) ──
print("\nStep: Normalization (zero mean, unit std)...")
l4_mean = np.nanmean(l4_log)
l4_std  = np.nanstd(l4_log)
l4_norm = (l4_log - l4_mean) / l4_std
print(f"Mean (before norm): {l4_mean:.4f}")
print(f"Std  (before norm): {l4_std:.4f}")

# ── Step 5: Save (only saving composites we actually have data for) ──
os.makedirs('data/l4_chl', exist_ok=True)
np.save('data/l4_chl/chl_l4_data.npy', l4_data)
np.save('data/l4_chl/chl_l4_log.npy',  l4_log)
np.save('data/l4_chl/chl_l4_norm.npy', l4_norm)
np.save('data/l4_chl/chl_l4_mean.npy', np.array([l4_mean]))
np.save('data/l4_chl/chl_l4_std.npy',  np.array([l4_std]))
print("\nSaved -> data/l4_chl/chl_l4_{data,log,norm,mean,std}.npy")
