"""
Seasonal Breakdown -- TEST SPLIT (leakage fix)
=================================================
Identical diffusion_pipeline() and season logic to
src/seasonal_chl_rmse.py. kappa=0.4, n_diff=50 -- unchanged winner.
Only change: obs_mask excludes the FULL held_out_20 (same as
before), RMSE scored only against test_mask.
"""
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

KAPPA, N_DIFF = 0.4, 50

chl_norm = np.load('data/modis_chl/chl_norm.npy')
dates    = np.load('data/modis_chl/chl_dates.npy')
N, LAT, LON = chl_norm.shape

split = np.load('results/holdout_split_masks.npz')
held_out_20 = split['held_out_20']
test_mask   = split['test_mask']
eval_t      = list(split['eval_t'])

def parse_date(ds):
    return datetime.strptime(str(ds), '%Y%m%d')
parsed = [parse_date(d) for d in dates]
months = np.array([d.month for d in parsed])

def get_season(month):
    if month in [6, 7, 8, 9]:   return 'SW Monsoon (Jun-Sep)'
    if month in [10,11,12,1]:   return 'NE Monsoon (Oct-Jan)'
    return 'Pre/Post-monsoon (Feb-May)'

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

season_results = {'SW Monsoon (Jun-Sep)': {'bg': [], 'pipe': [], 'cov': []},
                  'NE Monsoon (Oct-Jan)': {'bg': [], 'pipe': [], 'cov': []},
                  'Pre/Post-monsoon (Feb-May)': {'bg': [], 'pipe': [], 'cov': []}}

for t in eval_t:
    fc = chl_norm[t]; fp = chl_norm[t-1]
    valid_mask = ~np.isnan(fc)
    obs_mask = valid_mask & (~held_out_20[t])
    tm = test_mask[t]
    mu_b = fp.copy(); mu_b[np.isnan(mu_b)] = 0.0
    y_obs = np.where(obs_mask, fc, 0.0)
    mu_a = diffusion_pipeline(mu_b, y_obs, obs_mask)

    rows, cols = np.where(tm)
    if len(rows) == 0:
        continue
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
print(f"\n[Compare to original: SW=82.7%, NE=81.6%, Pre/Post=82.5%, Overall=82.3%]")

np.savez('results/seasonal_chl_rmse_testsplit.npz',
         seasons=list(season_results.keys()), results=str(season_results))
print("\nSaved -> results/seasonal_chl_rmse_testsplit.npz")
print("Done.")
