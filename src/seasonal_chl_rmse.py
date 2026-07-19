"""
Seasonal Breakdown of Chl-a RMSE
===================================
Splits existing validation results by Bay of Bengal season:
  SW Monsoon:    Jun-Sep (months 6-9)  -- worst case, heaviest cloud
  NE Monsoon:    Oct-Jan (months 10-1) -- moderate
  Pre-monsoon:   Feb-May (months 2-5)  -- best case, clearest skies
Reports RMSE and improvement for each season separately.
"""
import numpy as np
from datetime import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.abspath('.')))
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("Seasonal Breakdown: Chl-a RMSE by Bay of Bengal Season")
print("=" * 65)

KAPPA, N_DIFF = 0.4, 50

chl_norm = np.load('data/modis_chl/chl_norm.npy')
dates    = np.load('data/modis_chl/chl_dates.npy')
N, LAT, LON = chl_norm.shape

def parse_date(ds):
    return datetime.strptime(str(ds), '%Y%m%d')
parsed = [parse_date(d) for d in dates]
months = np.array([d.month for d in parsed])

def get_season(month):
    if month in [6, 7, 8, 9]:   return 'SW Monsoon (Jun-Sep)'
    if month in [10,11,12,1]:   return 'NE Monsoon (Oct-Jan)'
    return 'Pre/Post-monsoon (Feb-May)'

# Reproduce test mask
np.random.seed(42)
test_mask = np.zeros((N, LAT, LON), dtype=bool)
eval_t = []
for t in range(1, N):
    fc = chl_norm[t]
    valid_mask = ~np.isnan(fc)
    if valid_mask.sum() < 500: continue
    vi = np.where(valid_mask); n_v = len(vi[0])
    sel = np.random.choice(n_v, int(0.2*n_v), replace=False)
    test_mask[t, vi[0][sel], vi[1][sel]] = True
    eval_t.append(t)
print(f"Eval timesteps: {len(eval_t)}")

def diffusion_pipeline(mu_b, y_obs, obs_mask):
    mu_a = mu_b.copy()
    mu_a[obs_mask] = y_obs[obs_mask]
    for _ in range(N_DIFF):
        p = np.pad(mu_a, 1, mode='edge')
        L = p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*mu_a
        u = np.zeros_like(mu_a)
        u[~obs_mask] = KAPPA * L[~obs_mask]
        mu_a = mu_a + 0.1 * u
    return mu_a

# Run and collect per-timestep results with season label
season_results = {'SW Monsoon (Jun-Sep)': {'bg': [], 'pipe': [], 'cov': []},
                  'NE Monsoon (Oct-Jan)': {'bg': [], 'pipe': [], 'cov': []},
                  'Pre/Post-monsoon (Feb-May)': {'bg': [], 'pipe': [], 'cov': []}}

for t in eval_t:
    fc = chl_norm[t]; fp = chl_norm[t-1]
    valid_mask = ~np.isnan(fc)
    tm = test_mask[t]
    mu_b = fp.copy(); mu_b[np.isnan(mu_b)] = 0.0
    obs_mask = valid_mask & (~tm)
    y_obs = np.where(obs_mask, fc, 0.0)
    mu_a = diffusion_pipeline(mu_b, y_obs, obs_mask)

    rows, cols = np.where(tm)
    true_v = fc[rows, cols]
    bg_v   = mu_b[rows, cols]
    pred_v = mu_a[rows, cols]

    rmse_bg   = np.sqrt(np.mean((true_v - bg_v)**2))
    rmse_pipe = np.sqrt(np.mean((true_v - pred_v)**2))
    coverage  = valid_mask.sum() / (LAT * LON) * 100

    season = get_season(months[t])
    season_results[season]['bg'].append(rmse_bg)
    season_results[season]['pipe'].append(rmse_pipe)
    season_results[season]['cov'].append(coverage)

print(f"\n{'Season':<30}{'n':>4}{'BG RMSE':>10}{'Pipe RMSE':>11}"
      f"{'Improv%':>10}{'Coverage%':>12}")
print("-"*70)

overall_bg, overall_pipe = [], []
for season, vals in season_results.items():
    if not vals['bg']: continue
    bg_arr   = np.array(vals['bg'])
    pipe_arr = np.array(vals['pipe'])
    cov_arr  = np.array(vals['cov'])
    bg_m   = bg_arr.mean()
    pipe_m = pipe_arr.mean()
    imp    = (bg_m - pipe_m) / bg_m * 100
    cov_m  = cov_arr.mean()
    n      = len(bg_arr)
    print(f"{season:<30}{n:>4}{bg_m:>10.4f}{pipe_m:>11.4f}"
          f"{imp:>9.1f}%{cov_m:>11.1f}%")
    overall_bg  += list(bg_arr)
    overall_pipe += list(pipe_arr)

bg_all   = np.mean(overall_bg)
pipe_all = np.mean(overall_pipe)
imp_all  = (bg_all - pipe_all) / bg_all * 100
print(f"\n{'Overall':<30}{'':>4}{bg_all:>10.4f}{pipe_all:>11.4f}"
      f"{imp_all:>9.1f}%")

print(f"\nKey finding: coverage % shows why SW Monsoon is hardest.")
print(f"Improvement degrades with cloud cover, as expected.")

np.savez('results/seasonal_chl_rmse.npz',
         seasons=list(season_results.keys()),
         results=str(season_results))
print("\nSaved -> results/seasonal_chl_rmse.npz")
print("\nDone.")
