"""
Study Region Context Map -- Bay of Bengal, with real Chl-a data overlay
==========================================================================
Matches Mam's earlier paper's Fig. 1 convention:
  Gray  = land (from real Natural Earth coastline data)
  Green/blue shading = region with satellite data (real MODIS Chl-a values)
  White = missing data area (ocean, cloud-covered on the shown date, OR
          simply outside our downloaded study domain)

A red box marks our exact study domain. Land/ocean coloring comes
directly from cartopy's real geographic features -- no separate
categorical mask needed, since land/ocean is already correct
geography, and "missing" falls out naturally from NaN pixels in our
own data (or the complete absence of data outside our download box).
"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

print("="*65)
print("Study Region Context Map -- Bay of Bengal, with data overlay")
print("="*65)

chl_data = np.load('data/modis_chl/chl_data.npy')   # raw mg/m3
dates    = np.load('data/modis_chl/chl_dates.npy')
lat      = np.load('data/modis_chl/chl_lat.npy')
lon      = np.load('data/modis_chl/chl_lon.npy')

target_date = '20150101'
date_strs = [str(d) for d in dates]
idx = date_strs.index(target_date)
print(f"Target date: {dates[idx]} (composite index {idx})")

chl_log = np.log10(chl_data[idx])
n_valid = (~np.isnan(chl_log)).sum()
print(f"Valid Chl-a pixels on this date: {n_valid}")

study_lon = [80, 100]
study_lat = [5, 22]
wide_lon = [60, 110]
wide_lat = [-10, 35]

fig = plt.figure(figsize=(10, 9))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([wide_lon[0], wide_lon[1], wide_lat[0], wide_lat[1]], crs=ccrs.PlateCarree())

# Land = gray, Ocean = white (matches "missing data" -- both cloud
# gaps within our domain and the absence of data outside it)
ax.add_feature(cfeature.LAND, facecolor='#8c8c8c', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='#ffffff', zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=':', zorder=2)

# Real Chl-a data overlay, only where valid, only within our domain
im = ax.pcolormesh(lon, lat, chl_log, cmap='YlGnBu_r', vmin=-1, vmax=1.5,
                    transform=ccrs.PlateCarree(), zorder=1, shading='auto')

# Red box marking our exact study domain
box_lon = [study_lon[0], study_lon[1], study_lon[1], study_lon[0], study_lon[0]]
box_lat = [study_lat[0], study_lat[0], study_lat[1], study_lat[1], study_lat[0]]
ax.plot(box_lon, box_lat, color='red', linewidth=2, transform=ccrs.PlateCarree(), zorder=3)

gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.25, linestyle='--')
gl.top_labels = False
gl.right_labels = False

chennai_lon, chennai_lat = 80.27, 13.08
ax.plot(chennai_lon, chennai_lat, 'o', color='red', markersize=5,
        transform=ccrs.PlateCarree(), zorder=4)
ax.text(chennai_lon - 0.6, chennai_lat, 'Chennai', color='red', fontsize=10,
        fontweight='bold', transform=ccrs.PlateCarree(), zorder=4, va='center',
        ha='right')

cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label(r'$\log_{10}$(Chl-a, mg/m$^3$)', fontsize=10)

ax.set_title(f'Study region: Bay of Bengal (80-100°E, 5-22°N)\n'
             f'MODIS Chl-a, {dates[idx][:4]}-{dates[idx][4:6]}-{dates[idx][6:]}',
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('results/study_region_context_map.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved -> results/study_region_context_map.png")
