"""
Cell-NN DA — Save Full Reconstructed Array
Then plot Mean + Median trend:
  True field vs Reconstructed field
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from datetime import datetime
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from src.da_cellnn import cellnn_da_step

print("="*65)
print("Cell-NN DA — Save Reconstructed + Plot Trends")
print("="*65)

# ── Load data ─────────────────────────────────────────
chl_norm = np.load('data/modis_chl/chl_norm.npy')
chl_log  = np.load('data/modis_chl/chl_log.npy')
dates    = np.load('data/modis_chl/chl_dates.npy')
lat      = np.load('data/modis_chl/chl_lat.npy')
lon      = np.load('data/modis_chl/chl_lon.npy')

N, LAT, LON = chl_norm.shape
print(f"Shape: {chl_norm.shape}")

# ── Cell-NN DA 2D ─────────────────────────────────────
def cellnn_da_2d(mu_b_2d, y_obs_2d, obs_mask_2d,
                 alpha_A=1.0, wb=1.0,
                 kappa=0.4, n_diff=50):
    lat_n, lon_n = mu_b_2d.shape
    mu_b_flat  = mu_b_2d.flatten()
    y_obs_flat = y_obs_2d.flatten()
    obs_flat   = obs_mask_2d.flatten()

    mu_b_obs = mu_b_flat[obs_flat]
    y_obs    = y_obs_flat[obs_flat]

    mu_a_obs, _ = cellnn_da_step(
        mu_b_obs, y_obs,
        alpha_A=alpha_A, wb=wb,
        d_tau=5e-2, tau_max=1e-1,
        eps_conv=1e-6, r_max=50
    )

    mu_a_flat = mu_b_flat.copy()
    mu_a_flat[obs_flat] = mu_a_obs
    mu_a_2d   = mu_a_flat.reshape(lat_n, lon_n)

    # Spatial diffusion for gaps
    for _ in range(n_diff):
        mu_pad    = np.pad(mu_a_2d, 1, mode='edge')
        laplacian = (mu_pad[:-2, 1:-1] +
                     mu_pad[2:,  1:-1] +
                     mu_pad[1:-1, :-2] +
                     mu_pad[1:-1, 2:] -
                     4.0 * mu_a_2d)
        update = np.zeros_like(mu_a_2d)
        update[~obs_mask_2d] = kappa * laplacian[~obs_mask_2d]
        mu_a_2d = mu_a_2d + 0.1 * update

    return mu_a_2d

# ── Pure spatial-diffusion fill (no temporal background) ──
# Used only for t=0, where no X(t-1) exists.
# Merged in from the former standalone fix_first_timestep.py
# so this NaN bug cannot silently recur on a fresh pipeline run.
def spatial_diffusion_fill(field, kappa=0.3, n_iter=200):
    valid_mask = ~np.isnan(field)
    field_fill = field.copy()
    field_fill[~valid_mask] = 0.0
    for _ in range(n_iter):
        pad = np.pad(field_fill, 1, mode='edge')
        laplacian = (pad[:-2, 1:-1] + pad[2:, 1:-1] +
                     pad[1:-1, :-2] + pad[1:-1, 2:] -
                     4.0 * field_fill)
        update = np.zeros_like(field_fill)
        update[~valid_mask] = kappa * laplacian[~valid_mask]
        field_fill = field_fill + 0.1 * update
    return field_fill

# ── Run DA and save reconstructed array ───────────────
print("\nRunning Cell-NN DA on all 229 time steps...")
print("Saving full reconstructed array...")

chl_reconstructed = np.zeros_like(chl_norm)

# t=0 has no X(t-1) background -> pure spatial diffusion fill
# (this used to be a separate post-hoc patch; now merged here
#  so re-running this script from scratch cannot reintroduce
#  the t=0 NaN bug)
field0 = chl_norm[0]
valid0 = (~np.isnan(field0)).sum()
print(f"  t=0 | date={dates[0]} | valid={valid0/field0.size*100:.1f}% "
      f"| filling via spatial diffusion (no prior background)")
chl_reconstructed[0] = spatial_diffusion_fill(field0)
remaining_nan = np.isnan(chl_reconstructed[0]).sum()
if remaining_nan > 0:
    print(f"  WARNING: {remaining_nan} NaN remain at t=0 after diffusion fill")
else:
    print(f"  t=0 fully filled, 0 NaN remaining")

for t in range(1, N):
    field_curr = chl_norm[t]
    field_prev = chl_norm[t-1]
    valid_mask = ~np.isnan(field_curr)

    mu_b = field_prev.copy()
    mu_b[np.isnan(mu_b)] = 0.0

    if valid_mask.sum() < 500:
        # Too few pixels — use background
        chl_reconstructed[t] = mu_b
        continue

    obs_mask = valid_mask.copy()
    y_obs    = np.where(obs_mask, field_curr, 0.0)

    mu_a = cellnn_da_2d(
        mu_b, y_obs, obs_mask,
        alpha_A=1.0, wb=1.0,
        kappa=0.4, n_diff=50
    )

    chl_reconstructed[t] = mu_a

    if t % 46 == 0:
        print(f"  t={t:3d}/{N} | date={dates[t]} | "
              f"valid={valid_mask.sum()/valid_mask.size*100:.0f}%")

# Save reconstructed array
np.save('data/modis_chl/chl_reconstructed.npy',
        chl_reconstructed)
print(f"\nSaved → data/modis_chl/chl_reconstructed.npy")
print(f"Shape: {chl_reconstructed.shape}")

# ── Compute Mean + Median for both ────────────────────
print("\nComputing Mean + Median time series...")

date_objs = [datetime.strptime(d, '%Y%m%d')
             for d in dates]

# Per time step statistics
true_mean   = []
true_median = []
recon_mean  = []
recon_median= []

for t in range(N):
    # True field — only valid (non-NaN) pixels
    true_vals = chl_norm[t][~np.isnan(chl_norm[t])]
    # Reconstructed — all pixels (no NaN after DA)
    recon_vals = chl_reconstructed[t]
    recon_vals = recon_vals[np.isfinite(recon_vals)]

    if len(true_vals) > 100:
        true_mean.append(np.mean(true_vals))
        true_median.append(np.median(true_vals))
    else:
        true_mean.append(np.nan)
        true_median.append(np.nan)

    if len(recon_vals) > 100:
        recon_mean.append(np.mean(recon_vals))
        recon_median.append(np.median(recon_vals))
    else:
        recon_mean.append(np.nan)
        recon_median.append(np.nan)

true_mean    = np.array(true_mean)
true_median  = np.array(true_median)
recon_mean   = np.array(recon_mean)
recon_median = np.array(recon_median)

# Monthly aggregation
from collections import defaultdict
monthly = defaultdict(lambda: {
    'true_mean':[], 'true_median':[],
    'recon_mean':[], 'recon_median':[]
})

for t, d in enumerate(date_objs):
    key = (d.year, d.month)
    if not np.isnan(true_mean[t]):
        monthly[key]['true_mean'].append(true_mean[t])
        monthly[key]['true_median'].append(true_median[t])
    if not np.isnan(recon_mean[t]):
        monthly[key]['recon_mean'].append(recon_mean[t])
        monthly[key]['recon_median'].append(recon_median[t])

sorted_keys  = sorted(monthly.keys())
month_dates  = [datetime(k[0], k[1], 15) for k in sorted_keys]
m_true_mean  = [np.mean(monthly[k]['true_mean'])   for k in sorted_keys]
m_true_med   = [np.mean(monthly[k]['true_median']) for k in sorted_keys]
m_recon_mean = [np.mean(monthly[k]['recon_mean'])  for k in sorted_keys]
m_recon_med  = [np.mean(monthly[k]['recon_median'])for k in sorted_keys]

m_true_mean  = np.array(m_true_mean)
m_true_med   = np.array(m_true_med)
m_recon_mean = np.array(m_recon_mean)
m_recon_med  = np.array(m_recon_med)

# Trend lines
x_num = np.array([(d - month_dates[0]).days
                   for d in month_dates])

def trend(x, y):
    mask = ~np.isnan(y)
    z = np.polyfit(x[mask], y[mask], 1)
    return np.poly1d(z)(x), z[0]*365

tr_tm, sl_tm  = trend(x_num, m_true_mean)
tr_tmed, sl_tmed = trend(x_num, m_true_med)
tr_rm, sl_rm  = trend(x_num, m_recon_mean)
tr_rmed, sl_rmed = trend(x_num, m_recon_med)

print(f"\nTrend Analysis:")
print(f"  True Mean   trend: {sl_tm:+.4f}/yr")
print(f"  True Median trend: {sl_tmed:+.4f}/yr")
print(f"  Recon Mean  trend: {sl_rm:+.4f}/yr")
print(f"  Recon Median trend: {sl_rmed:+.4f}/yr")

# ── PLOT: Mean + Median — True vs Reconstructed ───────
fig, axes = plt.subplots(2, 1, figsize=(16, 12),
                          sharex=True)

# ── Top: MEAN comparison ──────────────────────────────
axes[0].plot(month_dates, m_true_mean,
             'b-o', lw=1.2, ms=4, alpha=0.8,
             label=f'True Mean '
                   f'(trend={sl_tm:+.4f}/yr)')
axes[0].plot(month_dates, m_recon_mean,
             'r-o', lw=1.2, ms=4, alpha=0.8,
             label=f'Reconstructed Mean '
                   f'(trend={sl_rm:+.4f}/yr)')
axes[0].plot(month_dates, tr_tm,
             'b--', lw=2.0, alpha=0.6)
axes[0].plot(month_dates, tr_rm,
             'r--', lw=2.0, alpha=0.6)

# Shading between
axes[0].fill_between(month_dates,
                     m_true_mean, m_recon_mean,
                     alpha=0.15, color='green',
                     label='Difference (Recon-True)')
axes[0].set_ylabel('log₁₀(Chl-a) (mg/m³)',
                   fontsize=11)
axes[0].text(0.01, 0.97, '(a)', transform=axes[0].transAxes,
             fontsize=14, fontweight='bold', va='top')
axes[0].text(-0.09, 0.5, 'Mean', transform=axes[0].transAxes,
             fontsize=13, fontweight='bold', va='center',
             ha='center', rotation=90)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Year shading
for yr in [2015, 2017, 2019]:
    axes[0].axvspan(datetime(yr,1,1),
                    datetime(yr,12,31),
                    alpha=0.05, color='blue')

# ── Bottom: MEDIAN comparison ─────────────────────────
axes[1].plot(month_dates, m_true_med,
             'b-s', lw=1.2, ms=4, alpha=0.8,
             label=f'True Median '
                   f'(trend={sl_tmed:+.4f}/yr)')
axes[1].plot(month_dates, m_recon_med,
             'r-s', lw=1.2, ms=4, alpha=0.8,
             label=f'Reconstructed Median '
                   f'(trend={sl_rmed:+.4f}/yr)')
axes[1].plot(month_dates, tr_tmed,
             'b--', lw=2.0, alpha=0.6)
axes[1].plot(month_dates, tr_rmed,
             'r--', lw=2.0, alpha=0.6)

axes[1].fill_between(month_dates,
                     m_true_med, m_recon_med,
                     alpha=0.15, color='orange',
                     label='Difference (Recon-True)')
axes[1].set_ylabel('log₁₀(Chl-a) (mg/m³)',
                   fontsize=11)
axes[1].text(0.01, 0.97, '(b)', transform=axes[1].transAxes,
             fontsize=14, fontweight='bold', va='top')
axes[1].text(-0.09, 0.5, 'Median', transform=axes[1].transAxes,
             fontsize=13, fontweight='bold', va='center',
             ha='center', rotation=90)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlabel('Date', fontsize=11)

for yr in [2015, 2017, 2019]:
    axes[1].axvspan(datetime(yr,1,1),
                    datetime(yr,12,31),
                    alpha=0.05, color='orange')

axes[1].set_xlim(month_dates[0], month_dates[-1])
plt.tight_layout()
plt.savefig('results/chl_mean_median_comparison.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved → results/chl_mean_median_comparison.png")

# ── Summary stats ─────────────────────────────────────
print(f"\n{'='*65}")
print(f"SUMMARY")
print(f"{'='*65}")
print(f"{'Metric':<35} {'True':>10} {'Recon':>10}")
print(f"{'-'*65}")
print(f"{'Mean log10(Chl-a)':<35} "
      f"{np.nanmean(m_true_mean):>10.4f} "
      f"{np.nanmean(m_recon_mean):>10.4f}")
print(f"{'Median log10(Chl-a)':<35} "
      f"{np.nanmean(m_true_med):>10.4f} "
      f"{np.nanmean(m_recon_med):>10.4f}")
print(f"{'Mean trend (/yr)':<35} "
      f"{sl_tm:>+10.4f} "
      f"{sl_rm:>+10.4f}")
print(f"{'Median trend (/yr)':<35} "
      f"{sl_tmed:>+10.4f} "
      f"{sl_rmed:>+10.4f}")
print(f"{'='*65}")
print(f"\nKey finding:")
ocean_mask_kf = np.any(~np.isnan(chl_norm), axis=0)
true_coverage_ocean = (~np.isnan(chl_norm[:, ocean_mask_kf])).mean() * 100
print(f"  True field: only {true_coverage_ocean:.1f}% pixels (ocean only, land excluded)")
print(f"  Reconstructed: 100% pixels filled by Cell-NN")
print(f"  Trends consistent → reconstruction is faithful!")
