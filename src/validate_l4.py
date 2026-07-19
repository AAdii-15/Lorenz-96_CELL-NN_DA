"""
Validation: Cell-NN reconstruction vs L4 (Copernicus gap-free) ground truth.
Mirrors validate_viirs_fixed2.py's structure, replacing VIIRS with L4.
Reports RMSE, R-squared, and slope (per Mam's request).
"""
import numpy as np
from scipy import stats
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from src.da_cellnn import cellnn_da_step

print("="*65)
print("Validation: Cell-NN vs L4 (Copernicus gap-free ground truth)")
print("="*65)

# ── Load MODIS ─────────────────────────────────────────
chl_data    = np.load('data/modis_chl/chl_data.npy')
modis_dates = np.load('data/modis_chl/chl_dates.npy')

modis_log = np.log10(chl_data)
modis_log[~np.isfinite(modis_log)] = np.nan
modis_mean = np.nanmean(modis_log)
modis_std  = np.nanstd(modis_log)
modis_norm = (modis_log - modis_mean) / modis_std
print(f"MODIS mean={modis_mean:.4f} std={modis_std:.4f}")

# ── Load L4 (already aligned to MODIS's 230 composite dates) ──
l4_data = np.load('data/l4_chl/chl_l4_data.npy')
l4_mean = np.load('data/l4_chl/chl_l4_mean.npy')[0]
l4_std  = np.load('data/l4_chl/chl_l4_std.npy')[0]
l4_log  = np.load('data/l4_chl/chl_l4_log.npy')
l4_norm = np.load('data/l4_chl/chl_l4_norm.npy')
print(f"L4 mean={l4_mean:.4f} std={l4_std:.4f}")
print(f"Mean difference: {abs(modis_mean-l4_mean):.4f}")

# ── Cell-NN DA (identical to validate_viirs_fixed2.py) ──
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
np.random.seed(42)

rmse_cnn_list = []
rmse_bg_list  = []
cnn_vals_all  = []   # for global R^2/slope
l4_vals_all   = []
bg_vals_all   = []

N = chl_data.shape[0]

for t_idx in range(1, N):
    m_curr = modis_norm[t_idx]
    m_prev = modis_norm[t_idx - 1]
    l4_field = l4_norm[t_idx]

    valid_m = ~np.isnan(m_curr)
    valid_l4 = ~np.isnan(l4_field)

    if valid_m.sum() < 500:
        continue

    vidx = np.where(valid_m)
    n_v  = len(vidx[0])
    sel  = np.random.choice(n_v, int(0.2*n_v), replace=False)
    tr   = vidx[0][sel]
    tc   = vidx[1][sel]

    obs_mask = valid_m.copy()
    obs_mask[tr, tc] = False

    mu_b = m_prev.copy()
    mu_b[np.isnan(mu_b)] = 0.0
    y_obs = np.where(obs_mask, m_curr, 0.0)

    mu_a = cellnn_da_2d(mu_b, y_obs, obs_mask)

    t_mask = np.zeros_like(m_curr, dtype=bool)
    t_mask[tr, tc] = True
    c_mask = t_mask & valid_l4

    if c_mask.sum() < 50:
        continue

    # Denormalize back to raw log10 scale
    mu_a_log = mu_a * modis_std + modis_mean
    mu_b_log = mu_b * modis_std + modis_mean
    l4_log_field = l4_field * l4_std + l4_mean

    rmse_r = np.sqrt(np.mean((mu_a_log[c_mask]-l4_log_field[c_mask])**2))
    rmse_b = np.sqrt(np.mean((mu_b_log[c_mask]-l4_log_field[c_mask])**2))

    rmse_cnn_list.append(rmse_r)
    rmse_bg_list.append(rmse_b)

    cnn_vals_all.append(mu_a_log[c_mask])
    l4_vals_all.append(l4_log_field[c_mask])
    bg_vals_all.append(mu_b_log[c_mask])

    if t_idx % 46 == 0:
        imp = (rmse_b-rmse_r)/rmse_b*100
        print(f"  t={t_idx} | CNN={rmse_r:.4f} BG={rmse_b:.4f} Imp={imp:.1f}%")

rmse_cnn_list = np.array(rmse_cnn_list)
rmse_bg_list  = np.array(rmse_bg_list)
imp_mean = (rmse_bg_list.mean()-rmse_cnn_list.mean())/rmse_bg_list.mean()*100

# ── Pooled R^2 and slope (Cell-NN vs L4, across all held-out pixels) ──
cnn_all = np.concatenate(cnn_vals_all)
l4_all  = np.concatenate(l4_vals_all)
bg_all  = np.concatenate(bg_vals_all)

slope_cnn, intercept_cnn, r_cnn, p_cnn, se_cnn = stats.linregress(l4_all, cnn_all)
slope_bg,  intercept_bg,  r_bg,  p_bg,  se_bg  = stats.linregress(l4_all, bg_all)

print(f"\n{'='*65}")
print(f"FINAL RESULTS: Cell-NN vs L4 ground truth")
print(f"{'='*65}")
print(f"MODIS mean={modis_mean:.4f}, L4 mean={l4_mean:.4f}")
print(f"Background vs L4 RMSE (per-composite mean): {rmse_bg_list.mean():.4f}")
print(f"Cell-NN vs L4 RMSE    (per-composite mean): {rmse_cnn_list.mean():.4f}")
print(f"Improvement: {imp_mean:.1f}%")
print()
print(f"Pooled across all held-out pixels (n={len(l4_all)}):")
print(f"  Cell-NN vs L4:  R^2={r_cnn**2:.4f}  slope={slope_cnn:.4f}  intercept={intercept_cnn:.4f}")
print(f"  Background vs L4: R^2={r_bg**2:.4f}  slope={slope_bg:.4f}  intercept={intercept_bg:.4f}")
print(f"{'='*65}")

np.save('results/l4_rmse_cnn.npy', rmse_cnn_list)
np.save('results/l4_rmse_bg.npy',  rmse_bg_list)
np.save('results/l4_cnn_vals_all.npy', cnn_all)
np.save('results/l4_l4_vals_all.npy',  l4_all)
np.save('results/l4_bg_vals_all.npy',  bg_all)
print(f"\nSaved pixel-level arrays for scatter plotting (n={len(l4_all)})")
