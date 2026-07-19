"""
Task 1: Save Cell-NN Reconstruction as .nc files
==================================================
Saves each 8-day reconstructed Chl-a field as a
NetCDF file with proper date/year naming.

Scale: log10(Chl-a) -- confirmed by Mam
Output: data/modis_chl/reconstructed_nc/
        AQUA_MODIS_CellNN.YYYYMMDD.nc
"""
import numpy as np
import xarray as xr
import os
from datetime import datetime

print("="*65)
print("Task 1: Save Cell-NN Reconstruction to .nc files")
print("Scale: log10(Chl-a) -- as confirmed by Mam")
print("="*65)

# ── Load data ─────────────────────────────────────────
chl_recon = np.load('data/modis_chl/chl_reconstructed.npy')  # log10-normalized scale
chl_norm  = np.load('data/modis_chl/chl_norm.npy')           # original normalized (with NaN)
dates     = np.load('data/modis_chl/chl_dates.npy')
lat       = np.load('data/modis_chl/chl_lat.npy')
lon       = np.load('data/modis_chl/chl_lon.npy')
chl_mean  = np.load('data/modis_chl/chl_mean.npy')[0]
chl_std   = np.load('data/modis_chl/chl_std.npy')[0]

print(f"Reconstruction shape: {chl_recon.shape}")
print(f"Dates: {dates[0]} to {dates[-1]}")
print(f"MODIS normalization params: mean={chl_mean:.4f}, std={chl_std:.4f}")

# ── Convert normalized -> log10(Chl-a) scale ──────────
# Mam confirmed: keep everything in log10(Chl-a) scale
# (NOT raw mg/m3, NOT z-score normalized)
chl_recon_log10 = chl_recon * chl_std + chl_mean   # back to log10(mg/m3)

print(f"\nReconstructed log10(Chl-a) range: "
      f"[{chl_recon_log10.min():.3f}, {chl_recon_log10.max():.3f}]")

# ── Output directory ───────────────────────────────────
OUT_DIR = 'data/modis_chl/reconstructed_nc'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Save each time step as a separate .nc file ────────
print(f"\nSaving {len(dates)} files to {OUT_DIR}/ ...")

for t in range(len(dates)):
    date_str = str(dates[t])
    date_obj = datetime.strptime(date_str, '%Y%m%d')

    da = xr.DataArray(
        chl_recon_log10[t].astype(np.float32),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="log10_chlor_a",
        attrs={
            "long_name": "Cell-NN reconstructed log10(Chlorophyll-a)",
            "units": "log10(mg m^-3)",
            "source": "Aqua-MODIS L3 + Cell-NN Data Assimilation",
            "date": date_obj.strftime("%Y-%m-%d"),
        }
    )

    ds = da.to_dataset()
    ds.attrs["title"] = "Cell-NN Reconstructed Chlorophyll-a (Bay of Bengal)"
    ds.attrs["region"] = "80-100E, 5-22N"
    ds.attrs["period_start"] = date_obj.strftime("%Y-%m-%d")
    ds.attrs["background"] = "X(t-1) previous 8-day MODIS image"
    ds.attrs["model"] = "Cell-NN DA (alpha_A=1.0, wb=1.0)"
    ds.attrs["created"] = datetime.now().strftime("%Y-%m-%d")

    fname = f"AQUA_MODIS_CellNN.{date_obj.strftime('%Y%m%d')}.nc"
    fpath = os.path.join(OUT_DIR, fname)
    ds.to_netcdf(fpath)

    if t % 46 == 0:
        print(f"  Saved {t+1}/{len(dates)}: {fname}")

print(f"\n✅ All {len(dates)} files saved!")

# ── Verify ──────────────────────────────────────────────
saved_files = sorted(os.listdir(OUT_DIR))
print(f"\nVerification:")
print(f"  Files in directory: {len(saved_files)}")
print(f"  First file: {saved_files[0]}")
print(f"  Last file : {saved_files[-1]}")

# Quick check - reopen one file
test = xr.open_dataset(os.path.join(OUT_DIR, saved_files[0]))
print(f"\nSample file check ({saved_files[0]}):")
print(test)
test.close()

print("\nTask 1 COMPLETE!")
print(f"Output: {OUT_DIR}/")
