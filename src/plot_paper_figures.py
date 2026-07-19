"""
Plots matching reference paper (Xie et al., 2024)
Fig 7: Seasonal spatial maps
Fig 8: Time series of mean + median Chl-a
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

print("="*60)
print("Plotting Paper Figures (Xie et al. 2024 style)")
print("="*60)

# ── Load data ─────────────────────────────────────────
chl_data = np.load('data/modis_chl/chl_data.npy')   # raw (230,408,480)
chl_log  = np.load('data/modis_chl/chl_log.npy')    # log10
dates    = np.load('data/modis_chl/chl_dates.npy')
lat      = np.load('data/modis_chl/chl_lat.npy')
lon      = np.load('data/modis_chl/chl_lon.npy')

date_objs = [datetime.strptime(d, '%Y%m%d') for d in dates]
months    = np.array([d.month for d in date_objs])
years     = np.array([d.year  for d in date_objs])

print(f"Shape: {chl_data.shape}")
print(f"Period: {dates[0]} to {dates[-1]}")

# ══════════════════════════════════════════════════════
# PLOT 1: Fig.8 style — Time series Mean + Median
# ══════════════════════════════════════════════════════
print("\nPlot 1: Time series Mean + Median...")

# Monthly aggregation
unique_months = []
mean_monthly  = []
median_monthly= []

# Group by year-month
from collections import defaultdict
monthly_data = defaultdict(list)

for i, d in enumerate(date_objs):
    key = (d.year, d.month)
    valid_vals = chl_log[i][~np.isnan(chl_log[i])]
    if len(valid_vals) > 100:
        monthly_data[key].append(valid_vals)

for key in sorted(monthly_data.keys()):
    all_vals = np.concatenate(monthly_data[key])
    unique_months.append(datetime(key[0], key[1], 15))
    mean_monthly.append(np.mean(all_vals))
    median_monthly.append(np.median(all_vals))

unique_months  = np.array(unique_months)
mean_monthly   = np.array(mean_monthly)
median_monthly = np.array(median_monthly)

# Annual mean (pink dots like paper)
annual_mean = {}
for i, d in enumerate(date_objs):
    yr = d.year
    vals = chl_log[i][~np.isnan(chl_log[i])]
    if len(vals) > 100:
        if yr not in annual_mean:
            annual_mean[yr] = []
        annual_mean[yr].append(np.mean(vals))

ann_years = sorted(annual_mean.keys())
ann_means = [np.mean(annual_mean[y]) for y in ann_years]
ann_dates = [datetime(y, 7, 1) for y in ann_years]

# Trend line
x_num = np.array([(d - unique_months[0]).days
                   for d in unique_months])
z_mean   = np.polyfit(x_num, mean_monthly, 1)
z_median = np.polyfit(x_num, median_monthly, 1)
trend_mean   = np.poly1d(z_mean)(x_num)
trend_median = np.poly1d(z_median)(x_num)

# Plot — exactly like Fig.8
fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                          sharex=True)

# Mean time series
axes[0].plot(unique_months, mean_monthly,
             color='steelblue', lw=1.0, alpha=0.8,
             label='Monthly mean')
axes[0].plot(unique_months, trend_mean,
             'k-', lw=2.0,
             label=f'Trend: {z_mean[0]*365:.4f}/yr')
axes[0].scatter(ann_dates, ann_means,
                color='pink', s=60, zorder=5,
                edgecolors='red', lw=1,
                label='Annual mean')
axes[0].set_ylabel('log₁₀(Chl-a) (mg/m³)',
                   fontsize=11)
axes[0].set_title(
    'Time Series of Spatially Averaged Chl-a\n'
    'Bay of Bengal (80-100°E, 5-22°N) | 2015-2019',
    fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(-1.0, 0.5)

# Shade alternate years
for yr in [2015, 2017, 2019]:
    axes[0].axvspan(datetime(yr,1,1),
                    datetime(yr,12,31),
                    alpha=0.05, color='blue')

# Median time series
axes[1].plot(unique_months, median_monthly,
             color='darkorange', lw=1.0, alpha=0.8,
             label='Monthly median')
axes[1].plot(unique_months, trend_median,
             'k-', lw=2.0,
             label=f'Trend: {z_median[0]*365:.4f}/yr')
axes[1].scatter(ann_dates,
                [np.median(np.concatenate(
                    [chl_log[i][~np.isnan(chl_log[i])]
                     for i, d in enumerate(date_objs)
                     if d.year == y]))
                 for y in ann_years],
                color='lightcoral', s=60, zorder=5,
                edgecolors='red', lw=1,
                label='Annual median')
axes[1].set_ylabel('log₁₀(Chl-a) (mg/m³)',
                   fontsize=11)
axes[1].set_title('Median Time Series',
                  fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel('Date', fontsize=11)
axes[1].set_ylim(-1.0, 0.5)

for yr in [2015, 2017, 2019]:
    axes[1].axvspan(datetime(yr,1,1),
                    datetime(yr,12,31),
                    alpha=0.05, color='orange')

plt.tight_layout()
plt.savefig('results/chl_timeseries_mean_median.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved → results/chl_timeseries_mean_median.png")

# Print stats
print(f"\nTime Series Statistics:")
print(f"  Mean  log10(Chl-a): {mean_monthly.mean():.4f}")
print(f"  Median log10(Chl-a): {median_monthly.mean():.4f}")
print(f"  Mean trend : {z_mean[0]*365:.6f} /year")
print(f"  Median trend: {z_median[0]*365:.6f} /year")

# ══════════════════════════════════════════════════════
# PLOT 2: Fig.7 style — Seasonal Spatial Maps
# ══════════════════════════════════════════════════════
print("\nPlot 2: Seasonal spatial maps...")

seasons = {
    'Spring\n(MAM)' : [3, 4, 5],
    'Summer\n(JJA)' : [6, 7, 8],
    'Autumn\n(SON)' : [9, 10, 11],
    'Winter\n(DJF)' : [12, 1, 2]
}

fig, axes = plt.subplots(1, 4, figsize=(22, 7))

for ax, (season_name, mnths) in zip(axes,
                                     seasons.items()):
    # Seasonal mean — raw Chl-a (like paper Fig.7)
    mask = np.array([d.month in mnths
                     for d in date_objs])
    chl_season = np.nanmean(chl_data[mask], axis=0)

    # Use log10 for display
    chl_log_s  = np.log10(chl_season + 1e-10)
    chl_log_s[~np.isfinite(chl_log_s)] = np.nan

    im = ax.imshow(
        chl_log_s,
        extent=[lon.min(), lon.max(),
                lat.min(), lat.max()],
        origin='upper',
        cmap='jet',
        vmin=-1.0, vmax=1.5,
        aspect='auto'
    )

    # Season mean value
    valid = chl_season[~np.isnan(chl_season)]
    mean_val = np.mean(valid) if len(valid) > 0 else 0

    ax.set_title(f'{season_name}\n'
                 f'Mean={mean_val:.3f} mg/m³',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude (°E)', fontsize=10)
    ax.set_ylabel('Latitude (°N)', fontsize=10)

    # Grid lines
    ax.set_xticks([80, 85, 90, 95, 100])
    ax.set_yticks([5, 10, 15, 20])
    ax.grid(True, alpha=0.2, color='white')

    cbar = plt.colorbar(im, ax=ax, shrink=0.85,
                        pad=0.02)
    cbar.set_label('log₁₀(Chl-a) (mg/m³)',
                   fontsize=9)

fig.suptitle(
    'Seasonal Distribution of Sea Surface Chl-a\n'
    'Bay of Bengal (80-100°E, 5-22°N) | 2015-2019\n'
    '(Like Xie et al. 2024, Fig.7)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('results/chl_seasonal_spatial.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved → results/chl_seasonal_spatial.png")

# ══════════════════════════════════════════════════════
# PLOT 3: Year-wise annual mean maps (bonus!)
# ══════════════════════════════════════════════════════
print("\nPlot 3: Year-wise annual maps...")

fig, axes = plt.subplots(1, 5, figsize=(25, 6))

for ax, yr in zip(axes, [2015, 2016, 2017, 2018, 2019]):
    mask = years == yr
    chl_yr    = np.nanmean(chl_data[mask], axis=0)
    chl_log_y = np.log10(chl_yr + 1e-10)
    chl_log_y[~np.isfinite(chl_log_y)] = np.nan

    valid = chl_yr[~np.isnan(chl_yr)]
    mean_val = np.mean(valid)

    im = ax.imshow(
        chl_log_y,
        extent=[lon.min(), lon.max(),
                lat.min(), lat.max()],
        origin='upper',
        cmap='jet',
        vmin=-1.0, vmax=1.5,
        aspect='auto'
    )
    ax.set_title(f'{yr}\nMean={mean_val:.3f} mg/m³',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Longitude (°E)', fontsize=9)
    ax.set_ylabel('Latitude (°N)', fontsize=9)
    ax.set_xticks([80, 85, 90, 95, 100])
    ax.set_yticks([5, 10, 15, 20])
    ax.grid(True, alpha=0.2, color='white')
    plt.colorbar(im, ax=ax, shrink=0.85,
                 label='log₁₀(Chl-a)')

fig.suptitle(
    'Annual Mean Chl-a Distribution\n'
    'Bay of Bengal (80-100°E, 5-22°N)',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('results/chl_annual_maps.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved → results/chl_annual_maps.png")

print("\n" + "="*60)
print("ALL PLOTS COMPLETE!")
print("="*60)
print("Saved:")
print("  results/chl_timeseries_mean_median.png (Fig.8 style)")
print("  results/chl_seasonal_spatial.png       (Fig.7 style)")
print("  results/chl_annual_maps.png            (bonus)")
