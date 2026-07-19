"""
Fix Table 8 coverage %: ocean-pixels-only denominator, matching the
paper's own stated methodology (Section 2.1: "Coverage percentages
are computed over ocean pixels only, with land excluded"). RMSE and
improvement numbers are untouched -- only the coverage column.
"""
import numpy as np
from datetime import datetime

chl_norm = np.load('data/modis_chl/chl_norm.npy')
dates    = np.load('data/modis_chl/chl_dates.npy')
N, LAT, LON = chl_norm.shape

ocean_mask = np.any(~np.isnan(chl_norm), axis=0)
n_ocean = int(ocean_mask.sum())
print(f"Ocean pixels: {n_ocean} / {LAT*LON} ({100*n_ocean/(LAT*LON):.1f}% of grid)")

split = np.load('results/holdout_split_masks.npz')
eval_t = list(split['eval_t'])

def parse_date(ds):
    return datetime.strptime(str(ds), '%Y%m%d')
parsed = [parse_date(d) for d in dates]
months = np.array([d.month for d in parsed])

def get_season(month):
    if month in [6,7,8,9]: return 'SW Monsoon (Jun-Sep)'
    if month in [10,11,12,1]: return 'NE Monsoon (Oct-Jan)'
    return 'Pre/Post-monsoon (Feb-May)'

season_cov = {'SW Monsoon (Jun-Sep)': [], 'NE Monsoon (Oct-Jan)': [], 'Pre/Post-monsoon (Feb-May)': []}
all_cov = []
for t in eval_t:
    fc = chl_norm[t]
    valid_ocean = (~np.isnan(fc)) & ocean_mask
    cov = valid_ocean.sum() / n_ocean * 100
    season_cov[get_season(months[t])].append(cov)
    all_cov.append(cov)

print(f"\n{'Season':<30}{'Coverage % (ocean-only)':>26}")
for s, vals in season_cov.items():
    print(f"{s:<30}{np.mean(vals):>26.1f}")
print(f"{'Overall':<30}{np.mean(all_cov):>26.1f}")
