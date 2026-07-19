"""
Task 3: Preprocessing MODIS Chl-a
1. Log transformation
2. Handle NaN (missing data)
3. Normalization
4. Save preprocessed data
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

print("="*60)
print("Task 3: Preprocessing MODIS Chl-a")
print("="*60)

# ── Load saved data ───────────────────────────────────
print("Loading data...")
chl_data = np.load('data/modis_chl/chl_data.npy')   # (230, 408, 480)
lat      = np.load('data/modis_chl/chl_lat.npy')
lon      = np.load('data/modis_chl/chl_lon.npy')
dates    = np.load('data/modis_chl/chl_dates.npy')

print(f"Shape: {chl_data.shape}")
print(f"NaN%  (before): {100*np.isnan(chl_data).mean():.1f}%")

# ── Step 1: Log Transformation ────────────────────────
print("\nStep 1: Log10 transformation...")
chl_log = np.log10(chl_data)
# -inf where chl=0, nan where chl=nan → set both to nan
chl_log[~np.isfinite(chl_log)] = np.nan
print(f"Log range: {np.nanmin(chl_log):.3f} to {np.nanmax(chl_log):.3f}")

# ── Step 2: Normalization ─────────────────────────────
print("\nStep 2: Normalization (zero mean, unit std)...")
chl_mean = np.nanmean(chl_log)
chl_std  = np.nanstd(chl_log)
chl_norm = (chl_log - chl_mean) / chl_std
print(f"Mean (before norm): {chl_mean:.4f}")
print(f"Std  (before norm): {chl_std:.4f}")
print(f"Mean (after norm) : {np.nanmean(chl_norm):.6f}")
print(f"Std  (after norm) : {np.nanstd(chl_norm):.6f}")

# ── Step 3: NaN Statistics ────────────────────────────
print("\nStep 3: NaN analysis...")
nan_per_timestep = np.isnan(chl_norm).mean(axis=(1,2)) * 100
print(f"NaN% per timestep:")
print(f"  Min  : {nan_per_timestep.min():.1f}%")
print(f"  Max  : {nan_per_timestep.max():.1f}%")
print(f"  Mean : {nan_per_timestep.mean():.1f}%")
print(f"  Total: {np.isnan(chl_norm).mean()*100:.1f}%")

# ── Step 4: Save ──────────────────────────────────────
print("\nSaving preprocessed data...")
np.save('data/modis_chl/chl_log.npy',    chl_log)
np.save('data/modis_chl/chl_norm.npy',   chl_norm)
np.save('data/modis_chl/chl_mean.npy',   np.array([chl_mean]))
np.save('data/modis_chl/chl_std.npy',    np.array([chl_std]))
print("Saved!")

# ── Plot: Before vs After ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

idx = 0  # First time step: 2015-01-01

# Raw
im0 = axes[0][0].imshow(chl_data[idx],
    extent=[lon.min(),lon.max(),lat.min(),lat.max()],
    origin='upper', cmap='jet', aspect='auto')
axes[0][0].set_title('(a) Raw Chl-a (mg/m³)', fontsize=11)
plt.colorbar(im0, ax=axes[0][0])

# Log
im1 = axes[0][1].imshow(chl_log[idx],
    extent=[lon.min(),lon.max(),lat.min(),lat.max()],
    origin='upper', cmap='jet', vmin=-1, vmax=2,
    aspect='auto')
axes[0][1].set_title('(b) Log₁₀(Chl-a)', fontsize=11)
plt.colorbar(im1, ax=axes[0][1])

# Normalized
im2 = axes[0][2].imshow(chl_norm[idx],
    extent=[lon.min(),lon.max(),lat.min(),lat.max()],
    origin='upper', cmap='RdBu_r', vmin=-3, vmax=3,
    aspect='auto')
axes[0][2].set_title('(c) Normalized Chl-a', fontsize=11)
plt.colorbar(im2, ax=axes[0][2])

# Histograms
valid_raw  = chl_data[np.isfinite(chl_data)].flatten()
valid_log  = chl_log[np.isfinite(chl_log)].flatten()
valid_norm = chl_norm[np.isfinite(chl_norm)].flatten()

axes[1][0].hist(valid_raw,  bins=100, color='blue',
                alpha=0.7, edgecolor='none')
axes[1][0].set_title('Histogram of (a)', fontsize=11)
axes[1][0].set_xlabel('Chl-a (mg/m³)')
axes[1][0].set_yscale('log')
axes[1][0].grid(True, alpha=0.3)

axes[1][1].hist(valid_log,  bins=100, color='green',
                alpha=0.7, edgecolor='none')
axes[1][1].set_title('Histogram of (b)', fontsize=11)
axes[1][1].set_xlabel('log₁₀(Chl-a)')
axes[1][1].grid(True, alpha=0.3)
axes[1][1].axvline(x=chl_mean, color='red', lw=2,
                   label=f'mean={chl_mean:.3f}')
axes[1][1].legend()

axes[1][2].hist(valid_norm, bins=100, color='purple',
                alpha=0.7, edgecolor='none')
axes[1][2].set_title('Histogram of (c)', fontsize=11)
axes[1][2].set_xlabel('Normalized value')
axes[1][2].grid(True, alpha=0.3)
axes[1][2].axvline(x=0, color='red', lw=2,
                   label='mean=0')
axes[1][2].legend()

fig.suptitle('Preprocessing Pipeline: Raw → Log → Normalized\n'
             'MODIS Chl-a Bay of Bengal 2015-01-01',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/chl_preprocessing.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved → results/chl_preprocessing.png")

# ── NaN% over time plot ───────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
date_objs = [datetime.strptime(d, '%Y%m%d') for d in dates]
ax.plot(date_objs, nan_per_timestep, 'r-', lw=1.0)
ax.axhline(y=nan_per_timestep.mean(), color='blue',
           lw=1.5, linestyle='--',
           label=f'Mean={nan_per_timestep.mean():.1f}%')
ax.fill_between(date_objs, nan_per_timestep,
                alpha=0.3, color='red')
ax.set_title('Missing Data (NaN%) Over Time\n'
             'High NaN = Cloud Cover (Monsoon Season)',
             fontsize=12)
ax.set_ylabel('Missing Data %')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/chl_nan_timeseries.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved → results/chl_nan_timeseries.png")

print("\n" + "="*60)
print("PREPROCESSING SUMMARY")
print("="*60)
print(f"Input  : Raw Chl-a  range [{np.nanmin(chl_data):.3f}, {np.nanmax(chl_data):.3f}] mg/m³")
print(f"Step 1 : Log10      range [{np.nanmin(chl_log):.3f}, {np.nanmax(chl_log):.3f}]")
print(f"Step 2 : Normalized range [{np.nanmin(chl_norm):.3f}, {np.nanmax(chl_norm):.3f}]")
print(f"NaN%   : {np.isnan(chl_norm).mean()*100:.1f}% (missing = clouds)")
print(f"Saved  : data/modis_chl/chl_norm.npy")
print("\nTask 3 COMPLETE!")
