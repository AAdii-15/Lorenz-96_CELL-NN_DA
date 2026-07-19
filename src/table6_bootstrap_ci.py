"""
Table 6 bootstrap 95% CIs -- TEST SPLIT
==========================================
Reuses the per-timestep RMSE arrays already saved by each
*_testsplit.py script. No new pipeline runs -- purely a
percentile bootstrap over timesteps, matching the paper's
stated protocol (Section 2.3, "Bootstrap confidence intervals").
"""
import numpy as np

def bootstrap_ci(rmse_bg, rmse_method, n_boot=10000, seed=123):
    n = len(rmse_method)
    rng = np.random.default_rng(seed)
    boot_rmse = np.empty(n_boot)
    boot_imp = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        m = rmse_method[idx].mean()
        bgm = rmse_bg[idx].mean()
        boot_rmse[b] = m
        boot_imp[b] = (bgm - m) / bgm * 100
    rlo, rhi = np.percentile(boot_rmse, [2.5, 97.5])
    ilo, ihi = np.percentile(boot_imp, [2.5, 97.5])
    return rlo, rhi, ilo, ihi

cellnn = np.load('results/chl_pipeline_testsplit.npz')
oi = np.load('results/oi_baseline_testsplit.npz')
dineof = np.load('results/dineof_testsplit.npz')
mc_rmse = np.load('results/montecarlo_rmse_testsplit.npy')
mc_bg = np.load('results/montecarlo_rmse_bg_testsplit.npy')

print("Bootstrapping 95% CIs (n=10,000 resamples, timestep-level, 229 timesteps)...\n")

bg_all = cellnn['rmse_bg']
rng0 = np.random.default_rng(999)
bg_boot = np.array([rng0.choice(bg_all, len(bg_all), replace=True).mean() for _ in range(10000)])
bg_lo, bg_hi = np.percentile(bg_boot, [2.5, 97.5])
print(f"{'Background':<22} RMSE={bg_all.mean():.4f} [{bg_lo:.4f}, {bg_hi:.4f}]")

for name, rbg, rm in [
    ('OI', oi['rmse_bg'], oi['rmse_oi']),
    ('DINEOF', dineof['rmse_bg'], dineof['rmse_dineof']),
    ('Diffusion pipeline', cellnn['rmse_bg'], cellnn['rmse_cellnn']),
    ('Monte-Carlo', mc_bg, mc_rmse),
]:
    rmean = rm.mean()
    imean = (rbg.mean() - rmean) / rbg.mean() * 100
    rlo, rhi, ilo, ihi = bootstrap_ci(rbg, rm)
    print(f"{name:<22} RMSE={rmean:.4f} [{rlo:.4f}, {rhi:.4f}]  "
          f"Improvement={imean:.1f}% [{ilo:.1f}%, {ihi:.1f}%]")
