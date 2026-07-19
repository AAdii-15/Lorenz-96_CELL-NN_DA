"""
Save True State as .npy file
as requested by supervisor
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lorenz63 import integrate_lorenz63, generate_observations, SIGMA, BETA, RHO
from src.lorenz96 import integrate_lorenz96, make_initial_condition, generate_observations_96

print("Saving true states...")

# ── Lorenz-63 ──────────────────────────────────────
w_true_ic = np.array([1.0, 3.0, 5.0])
w_bg_ic   = np.array([1.1, 3.3, 5.5])
dt63      = 1e-3
N63       = 20000

t63, w_true63 = integrate_lorenz63(w_true_ic, (0, N63*dt63), dt63)
w_obs63       = generate_observations(w_true63, noise_level=0.05)

np.save('results/lorenz63_true_state.npy',   w_true63)
np.save('results/lorenz63_background_ic.npy', w_bg_ic)
np.save('results/lorenz63_observations.npy',  w_obs63)
np.save('results/lorenz63_time.npy',          t63)

print(f"Lorenz-63 true state saved: {w_true63.shape}")

# ── Lorenz-96 ──────────────────────────────────────
X0_true96 = make_initial_condition(40, 8.0, perturb=0.0, seed=42)
X0_bg96   = make_initial_condition(40, 8.0, perturb=0.1, seed=123)
dt96      = 1e-2
N96       = 5000

t96, X_true96 = integrate_lorenz96(X0_true96, (0, N96*dt96), dt96)
X_obs96, obs_idx = generate_observations_96(X_true96, noise_level=0.05, obs_every=2)

np.save('results/lorenz96_true_state.npy',    X_true96)
np.save('results/lorenz96_background_ic.npy', X0_bg96)
np.save('results/lorenz96_observations.npy',  X_obs96)
np.save('results/lorenz96_obs_indices.npy',   obs_idx)
np.save('results/lorenz96_time.npy',          t96)

print(f"Lorenz-96 true state saved: {X_true96.shape}")

# ── Verify ─────────────────────────────────────────
print("\nVerification — files saved:")
for f in os.listdir('results'):
    if f.endswith('.npy'):
        data = np.load(f'results/{f}')
        print(f"  {f}: shape={data.shape}")

print("\nAll true states saved successfully!")
