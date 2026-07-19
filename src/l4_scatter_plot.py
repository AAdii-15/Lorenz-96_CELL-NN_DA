"""
Predicted vs actual (L4 ground truth) scatter/density plots.
(a) Background vs L4, (b) Cell-NN reconstruction vs L4.
Uses hexbin density since n~3.6M points would be an unreadable
solid blob as a plain scatter.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("Loading pixel-level validation arrays...")
l4_all  = np.load('results/l4_l4_vals_all.npy')
cnn_all = np.load('results/l4_cnn_vals_all.npy')
bg_all  = np.load('results/l4_bg_vals_all.npy')
print(f"n = {len(l4_all)}")

# Recompute fit stats for annotation (matches validate_l4.py exactly)
slope_cnn, intercept_cnn, r_cnn, p_cnn, se_cnn = stats.linregress(l4_all, cnn_all)
slope_bg,  intercept_bg,  r_bg,  p_bg,  se_bg  = stats.linregress(l4_all, bg_all)

rmse_cnn = np.sqrt(np.mean((cnn_all - l4_all)**2))
rmse_bg  = np.sqrt(np.mean((bg_all  - l4_all)**2))

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

panels = [
    (bg_all, 'Background', slope_bg, intercept_bg, r_bg**2, rmse_bg, axes[0]),
    (cnn_all, 'Cell-NN Reconstruction', slope_cnn, intercept_cnn, r_cnn**2, rmse_cnn, axes[1]),
]

vmin, vmax = -2.0, 1.5  # log10(Chl-a) range, matches preprocess_chl.py convention

for i, (pred, label, slope, intercept, r2, rmse, ax) in enumerate(panels):
    hb = ax.hexbin(l4_all, pred, gridsize=80, cmap='viridis',
                    bins='log', extent=[vmin, vmax, vmin, vmax])

    # 1:1 reference line
    ax.plot([vmin, vmax], [vmin, vmax], 'w--', lw=1.5, alpha=0.8,
            label='1:1 (ideal)')

    # Fitted regression line
    x_fit = np.array([vmin, vmax])
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r-', lw=2,
            label=f'Fit: slope={slope:.3f}')

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal')
    ax.set_xlabel('L4 ground truth (log$_{10}$ Chl-a)', fontsize=11)
    ax.set_ylabel(f'{label} (log$_{{10}}$ Chl-a)', fontsize=11)
    ax.text(0.02, 0.98, ['(a)', '(b)'][i], transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', color='white')
    ax.text(0.98, 0.02,
             f'R$^2$={r2:.3f}\nRMSE={rmse:.3f}\nn={len(pred):,}',
             transform=ax.transAxes, fontsize=10, va='bottom', ha='right',
             color='white',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    ax.legend(loc='upper left', fontsize=9, framealpha=0.8)
    cb = plt.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label('log$_{10}$(count)', fontsize=9)

plt.tight_layout()
plt.savefig('results/l4_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved -> results/l4_predicted_vs_actual.png")

print(f"\nSanity check against validate_l4.py output:")
print(f"  Background: R^2={r_bg**2:.4f} slope={slope_bg:.4f} RMSE={rmse_bg:.4f}")
print(f"  Cell-NN:    R^2={r_cnn**2:.4f} slope={slope_cnn:.4f} RMSE={rmse_cnn:.4f}")
