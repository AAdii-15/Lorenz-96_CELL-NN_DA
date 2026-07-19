"""
Validation FINAL: Separate normalization for MODIS and VIIRS
Mam's feedback: normalize both datasets independently
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
import glob, os
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src.da_cellnn import cellnn_da_step

print("="*65)
print("Validation FINAL: Independent Normalization")
print("MODIS: own mean/std | VIIRS: own mean/std")
print("="*65)

# ── Load MODIS ─────────────────────────────────────────
chl_data      = np.load('data/modis_chl/chl_data.npy')
modis_dates   = np.load('data/modis_chl/chl_dates.npy')
lat = np.load('data/modis_chl/chl_lat.npy')
lon = np.load('data/modis_chl/chl_lon.npy')

# MODIS log + normalize with MODIS own stats
modis_log = np.log10(chl_data)
modis_log[~np.isfinite(modis_log)] = np.nan
modis_mean = np.nanmean(modis_log)
modis_std  = np.nanstd(modis_log)
modis_norm = (modis_log - modis_mean) / modis_std
print(f"MODIS mean={modis_mean:.4f} std={modis_std:.4f}")

# ── Load VIIRS ─────────────────────────────────────────
VIIRS_DIR   = "data/viirs_chl/nc_files"
viirs_files = sorted(glob.glob(f"{VIIRS_DIR}/*.nc"))

viirs_raw   = []
viirs_dates = []
for f in viirs_files:
    ds  = xr.open_dataset(f)
    viirs_raw.append(ds['chlor_a'].values)
    fname    = os.path.basename(f)
    viirs_dates.append(fname.split('.')[1].split('_')[0])
    ds.close()

viirs_raw   = np.array(viirs_raw)
viirs_dates = np.array(viirs_dates)

# VIIRS log + normalize with VIIRS own stats
viirs_log = np.log10(viirs_raw)
viirs_log[~np.isfinite(viirs_log)] = np.nan
viirs_mean = np.nanmean(viirs_log)
viirs_std  = np.nanstd(viirs_log)
viirs_norm = (viirs_log - viirs_mean) / viirs_std
print(f"VIIRS mean={viirs_mean:.4f} std={viirs_std:.4f}")
print(f"Mean difference: {abs(modis_mean-viirs_mean):.4f}")

# ── Cell-NN DA ─────────────────────────────────────────
def cellnn_da_2d(mu_b, y_obs, obs_mask,
                 alpha_A=1.0, wb=1.0,
                 kappa=0.4, n_diff=50):
    lat_n, lon_n = mu_b.shape
    mu_b_f  = mu_b.flatten()
    y_f     = y_obs.flatten()
    obs_f   = obs_mask.flatten()
    mu_a_obs, _ = cellnn_da_step(
        mu_b_f[obs_f], y_f[obs_f],
        alpha_A=alpha_A, wb=wb,
        d_tau=5e-2, tau_max=1e-1,
        eps_conv=1e-6, r_max=50)
    mu_a_f          = mu_b_f.copy()
    mu_a_f[obs_f]   = mu_a_obs
    mu_a            = mu_a_f.reshape(lat_n, lon_n)
    for _ in range(n_diff):
        p = np.pad(mu_a, 1, mode='edge')
        L = p[:-2,1:-1]+p[2:,1:-1]+p[1:-1,:-2]+p[1:-1,2:]-4*mu_a
        u = np.zeros_like(mu_a)
        u[~obs_mask] = kappa*L[~obs_mask]
        mu_a = mu_a + 0.1*u
    return mu_a

# ── Run Validation ─────────────────────────────────────
print("\nRunning validation...")
modis_date_set = {d: i for i,d in enumerate(modis_dates)}
viirs_date_set = {d: i for i,d in enumerate(viirs_dates)}
common_dates   = sorted(set(modis_dates)&set(viirs_dates))

np.random.seed(42)
rmse_cnn = []
rmse_bg  = []
dates_u  = []
saved    = []

for t_idx, date in enumerate(common_dates[1:], 1):
    mi      = modis_date_set[date]
    vi      = viirs_date_set[date]
    mi_prev = modis_date_set[common_dates[t_idx-1]]

    m_curr  = modis_norm[mi]
    m_prev  = modis_norm[mi_prev]
    v_field = viirs_norm[vi]

    valid_m = ~np.isnan(m_curr)
    valid_v = ~np.isnan(v_field)

    if valid_m.sum() < 500:
        continue

    # Mask 20% MODIS pixels
    vidx  = np.where(valid_m)
    n_v   = len(vidx[0])
    sel   = np.random.choice(n_v, int(0.2*n_v),
                              replace=False)
    tr    = vidx[0][sel]
    tc    = vidx[1][sel]

    obs_mask = valid_m.copy()
    obs_mask[tr, tc] = False

    mu_b = m_prev.copy()
    mu_b[np.isnan(mu_b)] = 0.0
    y_obs = np.where(obs_mask, m_curr, 0.0)

    mu_a = cellnn_da_2d(mu_b, y_obs, obs_mask)

    # Compare at masked pixels where VIIRS also valid
    t_mask = np.zeros_like(m_curr, dtype=bool)
    t_mask[tr, tc] = True
    c_mask = t_mask & valid_v

    if c_mask.sum() < 50:
        continue

    # Convert Cell-NN back to log10 scale
    # Then convert to VIIRS scale for fair comparison
    mu_a_log  = mu_a * modis_std + modis_mean
    mu_b_log  = mu_b * modis_std + modis_mean
    v_log     = v_field * viirs_std + viirs_mean

    rmse_r = np.sqrt(np.mean(
        (mu_a_log[c_mask]-v_log[c_mask])**2))
    rmse_b = np.sqrt(np.mean(
        (mu_b_log[c_mask]-v_log[c_mask])**2))

    rmse_cnn.append(rmse_r)
    rmse_bg.append(rmse_b)
    dates_u.append(datetime.strptime(date,'%Y%m%d'))

    if t_idx % 46 == 0:
        imp = (rmse_b-rmse_r)/rmse_b*100
        print(f"  t={t_idx} | {date} | "
              f"CNN={rmse_r:.4f} BG={rmse_b:.4f} "
              f"Imp={imp:.1f}%")

    if t_idx % 46 == 0 and len(saved) < 3:
        saved.append({
            'date':date, 'modis':m_curr,
            'viirs':v_field, 'bg':mu_b,
            'cnn':mu_a, 'rmse_r':rmse_r,
            'rmse_b':rmse_b
        })

rmse_cnn = np.array(rmse_cnn)
rmse_bg  = np.array(rmse_bg)
imp_mean = (rmse_bg.mean()-rmse_cnn.mean())/rmse_bg.mean()*100

print(f"\n{'='*65}")
print(f"FINAL RESULTS (Independent Normalization)")
print(f"{'='*65}")
print(f"MODIS stats: mean={modis_mean:.4f} std={modis_std:.4f}")
print(f"VIIRS stats: mean={viirs_mean:.4f} std={viirs_std:.4f}")
print(f"Background vs VIIRS RMSE: {rmse_bg.mean():.4f}")
print(f"Cell-NN vs VIIRS RMSE   : {rmse_cnn.mean():.4f}")
print(f"Improvement             : {imp_mean:.1f}%")
print(f"{'='*65}")

np.save('results/viirs_rmse_cnn_final.npy', rmse_cnn)
np.save('results/viirs_rmse_bg_final.npy',  rmse_bg)

# ── Plot ──────────────────────────────────────────────
fig, axes = plt.subplots(2,1,figsize=(16,10),sharex=True)

axes[0].plot(dates_u, rmse_bg,  'b-o', ms=3, lw=1.2,
             alpha=0.8,
             label=f'Background vs VIIRS '
                   f'RMSE={rmse_bg.mean():.4f}')
axes[0].plot(dates_u, rmse_cnn, 'r-o', ms=3, lw=1.2,
             alpha=0.8,
             label=f'Cell-NN vs VIIRS '
                   f'RMSE={rmse_cnn.mean():.4f}')
axes[0].set_title(
    'VIIRS Validation — Independent Normalization\n'
    'MODIS & VIIRS normalized independently',
    fontsize=12, fontweight='bold')
axes[0].set_ylabel('RMSE (log₁₀ scale)')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

imp_ts = (rmse_bg-rmse_cnn)/rmse_bg*100
axes[1].plot(dates_u, imp_ts, 'g-', lw=1.0)
axes[1].axhline(y=imp_ts.mean(), color='darkgreen',
                lw=2, linestyle='--',
                label=f'Mean={imp_ts.mean():.1f}%')
axes[1].fill_between(dates_u, 0, imp_ts,
                     where=imp_ts>0,
                     alpha=0.3, color='green',
                     label='Cell-NN better')
axes[1].fill_between(dates_u, 0, imp_ts,
                     where=imp_ts<0,
                     alpha=0.3, color='red',
                     label='BG better')
axes[1].axhline(y=0, color='k', lw=1)
axes[1].set_title('Improvement %', fontsize=12)
axes[1].set_ylabel('Improvement %')
axes[1].set_xlabel('Date')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/viirs_validation_final.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved → results/viirs_validation_final.png")

# ── Summary ───────────────────────────────────────────
print(f"\nPAPER RESULT (Final):")
print(f"  MODIS mean={modis_mean:.4f}, VIIRS mean={viirs_mean:.4f}")
print(f"  Cell-NN RMSE: {rmse_cnn.mean():.4f}")
print(f"  Background  : {rmse_bg.mean():.4f}")
print(f"  Improvement : {imp_mean:.1f}%")
