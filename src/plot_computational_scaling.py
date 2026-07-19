"""
Computational scaling figure: wall-clock time vs. state dimension K,
log-log scale. Uses results/computational_scaling.npz (K in
{10,20,40,80,160}, N=1000 steps, 1 repeat per point -- a lighter,
separate run from Table 5's K=40-only, N=5000, 3-repeat timings, built
specifically to show the scaling TREND, not to reproduce Table 5's
absolute numbers exactly).
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.load('results/computational_scaling.npz')
K_vals = data['K_vals']
t_3dvar = data['t_3dvar']
t_cellnn = data['t_cellnn']
t_enkf = data['t_enkf']

fig, ax = plt.subplots(figsize=(6, 5))

ax.loglog(K_vals, t_3dvar, 'o-', color='#d62728', label='3D-Var', lw=1.8, markersize=7)
ax.loglog(K_vals, t_cellnn, 's-', color='#1f77b4', label='Cell-NN', lw=1.8, markersize=7)
ax.loglog(K_vals, t_enkf, '^-', color='#2ca02c', label='EnKF (N=50)', lw=1.8, markersize=7)

ax.set_xlabel('State dimension $K$', fontsize=11)
ax.set_ylabel('Wall-clock time (s)', fontsize=11)
ax.set_title('Computational scaling with state dimension\n'
              '(N=1000 steps, 1 repeat per point)', fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, which='both', alpha=0.3)
ax.set_xticks(K_vals)
ax.set_xticklabels([str(k) for k in K_vals])

plt.tight_layout()
plt.savefig('results/computational_scaling.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved -> results/computational_scaling.png")
