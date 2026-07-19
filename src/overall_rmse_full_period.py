"""
Single Overall RMSE -- Full 2015-2019, log10 scale
Cell-NN Reconstructed (real, full deployment) vs Independent VIIRS
=========================================================================
Third of Mam's three pending tasks. Unlike the internal/external
validation numbers (which use a synthetic 20%-pixel-masking protocol),
this uses the ACTUAL deployed Cell-NN reconstruction (chl_reconstructed.npy,
built from ALL real MODIS observations, no held-out pixels) compared
against ALL valid VIIRS pixels across the full 5-year record, pooled
into a SINGLE RMSE number, in physically meaningful log10(Chl-a mg/m3)
units (not the abstract normalized z-score units used elsewhere).
"landmask" is not applied separately -- land pixels are already NaN
in both MODIS and VIIRS and excluded by standard NaN handling.
"""
import numpy as np
import xarray as xr
import glob, os

print("=" * 65)
print("Single Overall RMSE -- Full 2015-2019, log10 scale")
print("Cell-NN Reconstructed vs Independent VIIRS")
print("=" * 65)

chl_reconstructed = np.load('data/modis_chl/chl_reconstructed.npy')
chl_mean = np.load('data/modis_chl/chl_mean.npy')[0]
chl_std = np.load('data/modis_chl/chl_std.npy')[0]
modis_dates = np.load('data/modis_chl/chl_dates.npy')
modis_date_strs = [str(d) for d in modis_dates]

cellnn_log_all = chl_reconstructed * chl_std + chl_mean
print(f"Cell-NN reconstruction shape: {cellnn_log_all.shape}")
print(f"NaN in reconstruction: {int(np.isnan(cellnn_log_all).sum())} (should be 0)")

VIIRS_DIR = "data/viirs_chl/nc_files"
viirs_files = sorted(glob.glob(f"{VIIRS_DIR}/*.nc"))

viirs_log_by_date = {}
for f in viirs_files:
    fname = os.path.basename(f)
    file_date = fname.split('.')[1].split('_')[0]
    ds = xr.open_dataset(f)
    viirs_raw = ds['chlor_a'].values
    ds.close()
    viirs_log = np.log10(viirs_raw)
    viirs_log[~np.isfinite(viirs_log)] = np.nan
    viirs_log_by_date[file_date] = viirs_log

print(f"VIIRS files loaded: {len(viirs_log_by_date)}")

common_dates = sorted(set(modis_date_strs) & set(viirs_log_by_date.keys()))
print(f"Common dates (MODIS & VIIRS): {len(common_dates)}")

all_true, all_pred = [], []

for date in common_dates:
    mi = modis_date_strs.index(date)
    cellnn_field = cellnn_log_all[mi]
    viirs_field = viirs_log_by_date[date]

    valid_v = ~np.isnan(viirs_field)
    if valid_v.sum() == 0:
        continue

    all_true.append(viirs_field[valid_v])
    all_pred.append(cellnn_field[valid_v])

all_true = np.concatenate(all_true)
all_pred = np.concatenate(all_pred)

print(f"\nTotal pixel-instances pooled: {len(all_true):,}")

rmse_overall = np.sqrt(np.mean((all_true - all_pred) ** 2))
bias = np.mean(all_pred - all_true)
mae = np.mean(np.abs(all_pred - all_true))

print(f"\n{'=' * 65}")
print("SINGLE OVERALL RESULT -- Full 2015-2019, log10(Chl-a mg/m3) scale")
print(f"{'=' * 65}")
print(f"{'Metric':<30}{'Value':>15}")
print("-" * 65)
print(f"{'RMSE (log10 scale)':<30}{rmse_overall:>15.4f}")
print(f"{'MAE (log10 scale)':<30}{mae:>15.4f}")
print(f"{'Bias (mean error)':<30}{bias:>15.4f}")
print(f"{'n (pooled pixel-instances)':<30}{len(all_true):>15,}")
print(f"{'n (common dates)':<30}{len(common_dates):>15}")
print(f"{'=' * 65}")

np.savez('results/overall_rmse_full_period.npz',
         all_true=all_true, all_pred=all_pred,
         rmse=rmse_overall, mae=mae, bias=bias)
print("\nSaved -> results/overall_rmse_full_period.npz")
print("\nDone.")
