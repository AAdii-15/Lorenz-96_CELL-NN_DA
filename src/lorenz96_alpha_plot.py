"""
Lorenz-96 alpha_A optimization: 3-panel summary figure.
==========================================================
(a) R-score vs alpha_A, averaged over 5 seeds, with error bars
    (std across seeds) -- shows R-score is relatively FLAT and
    noisy across alpha_A, not a sharp minimum at any one value.
(b) Average relaxation iterations used per DA cycle vs alpha_A --
    shows the iteration cap (r_max=50) is binding for all
    alpha_A <~2.5, only easing for larger alpha_A.
(c) Single-relaxation-step error vs alpha_A (log scale) -- the
    same metric behind Figure 2, showing the theoretically-clean,
    monotonic convergence-rate improvement with alpha_A.
"""
import numpy as np
import matplotlib.pyplot as plt

results = np.load('results/lorenz96_alpha_sweep.npy', allow_pickle=True)
single_step = np.load('results/lorenz96_alpha_single_step.npy', allow_pickle=True)

alpha_vals = [r['alpha_A'] for r in results]
r_means = [r['r_score_mean'] for r in results]
r_stds  = [r['r_score_std'] for r in results]
avg_iters = [r['avg_iter'] for r in results]

alpha_vals_ss = [s['alpha_A'] for s in single_step]
errors_ss = [s['error'] for s in single_step]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# ── Panel (a): R-score vs alpha_A, with error bars ──
ax = axes[0]
ax.errorbar(alpha_vals, r_means, yerr=r_stds, fmt='o-', color='purple',
            capsize=4, lw=1.5, ms=6)
best_idx = np.argmin(r_means)
ax.axvline(alpha_vals[best_idx], color='gray', linestyle=':', alpha=0.6)
ax.axvline(1.0, color='green', linestyle='--', alpha=0.7, label='$\\alpha_A=1.0$ (used in paper)')
ax.set_xscale('log')
ax.set_xlabel('$\\alpha_A$', fontsize=12)
ax.set_ylabel('R-score (mean $\\pm$ std, 5 seeds)', fontsize=11)
ax.set_title('(a) DA Performance vs $\\alpha_A$\n(flat/noisy -- not a sharp minimum)',
              fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=13,
        fontweight='bold', va='top')

# ── Panel (b): Avg iterations vs alpha_A ──
ax = axes[1]
ax.plot(alpha_vals, avg_iters, 'o-', color='darkorange', lw=1.5, ms=6)
ax.axhline(50, color='gray', linestyle=':', alpha=0.6, label='$r_{max}=50$ (cap)')
ax.axvline(1.0, color='green', linestyle='--', alpha=0.7, label='$\\alpha_A=1.0$ (used in paper)')
ax.set_xscale('log')
ax.set_xlabel('$\\alpha_A$', fontsize=12)
ax.set_ylabel('Avg. relaxation iterations per DA cycle', fontsize=11)
ax.set_title('(b) Iteration Cap Binding vs $\\alpha_A$\n(cap binds for $\\alpha_A \\lesssim 2.5$)',
              fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=13,
        fontweight='bold', va='top')

# ── Panel (c): Single-relaxation-step error vs alpha_A ──
ax = axes[2]
ax.semilogy(alpha_vals_ss, errors_ss, 'o-', color='steelblue', lw=1.5, ms=6)
ax.axvline(1.0, color='green', linestyle='--', alpha=0.7, label='$\\alpha_A=1.0$ (used in paper)')
ax.set_xscale('log')
ax.set_xlabel('$\\alpha_A$', fontsize=12)
ax.set_ylabel('Single-step relaxation error (log scale)', fontsize=11)
ax.set_title('(c) Single-Step Convergence vs $\\alpha_A$\n(clean, monotonic -- matches Fig. 2)',
              fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')
ax.text(0.02, 0.98, '(c)', transform=ax.transAxes, fontsize=13,
        fontweight='bold', va='top')

plt.tight_layout()
plt.savefig('results/lorenz96_alpha_optimization.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved -> results/lorenz96_alpha_optimization.png")

# ── Print summary for reference ──
print(f"\nBest R-score: alpha_A={alpha_vals[best_idx]:.2f} (R={r_means[best_idx]:.4f})")
print(f"alpha_A=1.0 R-score: {r_means[alpha_vals.index(1.0)]:.4f}")
print(f"Difference: {abs(r_means[best_idx] - r_means[alpha_vals.index(1.0)]):.4f} "
      f"(within 1 std: {abs(r_means[best_idx] - r_means[alpha_vals.index(1.0)]) < r_stds[alpha_vals.index(1.0)]})")
