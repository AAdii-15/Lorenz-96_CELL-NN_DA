"""
Multi-region fixed-region test -- blind, pre-registered sample
====================================================================
De-risks the single-region result in rect_mask_final.py. Candidates
are drawn from a systematic grid, filtered only by ocean fraction
(>=70%, the same "mostly water" criterion originally specified,
well above the failed Andaman Sea candidate's ~39% ocean), sampled
with a fixed seed BEFORE any RMSE is computed, and the region already
used in Figure 11 (rows 190-250, cols 0-60) is explicitly excluded,
so this is an independent check, not a repeat. Single pass at the
production setting (kappa=0.4, n_diff=50) per region -- no per-region
stability sweep, matching the scope of a de-risking check rather than
a full re-derivation.
"""
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.da_cellnn import cellnn_da_step

print("="*70)
print("Multi-region fixed-region test -- blind pre-registered sample")
print("="*70)

chl_norm = np.load('data/modis_chl/chl_norm.npy')
chl_l4_norm = np.load('data/l4_chl/chl_l4_norm.npy')
dates = np.load('data/modis_chl/chl_dates.npy')
lat = np.load('data/modis_chl/chl_lat.npy')
lon = np.load('data/modis_chl/chl_lon.npy')

target_idx = [i for i, d in enumerate(dates) if str(d) == '20170101'][0]
print(f"Target date: {dates[target_idx]} (composite index {target_idx}), "
      f"fixed for all regions to isolate region choice as the only variable")

field_curr = chl_norm[target_idx]
field_prev = chl_norm[target_idx - 1]
l4_field = chl_l4_norm[target_idx]
valid_mask = ~np.isnan(field_curr)
ocean_mask = np.any(~np.isnan(chl_norm), axis=0)
land_mask = ~ocean_mask
LAT_N, LON_N = ocean_mask.shape

BOX = 60
USED_REGION = (190, 0)  # already in Figure 11 -- excluded from the blind sample
OCEAN_FRAC_MIN = 0.70
N_SAMPLE = 5
STRIDE = 80

# ── Step 1: build candidate grid, filter by ocean fraction ──
candidates = []
for r0 in range(0, LAT_N - BOX + 1, STRIDE):
    for c0 in range(0, LON_N - BOX + 1, STRIDE):
        if (r0, c0) == USED_REGION:
            continue
        ocean_frac = ocean_mask[r0:r0+BOX, c0:c0+BOX].mean()
        if ocean_frac >= OCEAN_FRAC_MIN:
            candidates.append((r0, c0, ocean_frac))

print(f"\nCandidate grid: stride={STRIDE}px, box={BOX}x{BOX}")
print(f"Candidates passing ocean-fraction >= {OCEAN_FRAC_MIN:.0%} filter: {len(candidates)}")

# ── Step 2: fixed-seed sample, BEFORE computing any RMSE ──
rng = np.random.RandomState(42)
sel_idx = rng.choice(len(candidates), size=min(N_SAMPLE, len(candidates)), replace=False)
sampled = [candidates[i] for i in sel_idx]

print(f"\nBlind sample (seed=42, chosen before any RMSE computed):")
for r0, c0, frac in sampled:
    print(f"  rows {r0}-{r0+BOX}, cols {c0}-{c0+BOX}, ocean={frac:.1%}")

def cellnn_da_2d(mu_b_2d, y_obs_2d, obs_mask_2d, alpha_A=1.0, wb=1.0, kappa=0.4, n_diff=50):
    lat_n, lon_n = mu_b_2d.shape
    mu_b_flat = mu_b_2d.flatten(); y_obs_flat = y_obs_2d.flatten(); obs_flat = obs_mask_2d.flatten()
    mu_b_obs = mu_b_flat[obs_flat]; y_obs = y_obs_flat[obs_flat]
    mu_a_obs, n_iter = cellnn_da_step(
        mu_b_obs, y_obs, alpha_A=alpha_A, wb=wb,
        d_tau=5e-2, tau_max=1e-1, eps_conv=1e-6, r_max=50)
    mu_a_flat = mu_b_flat.copy(); mu_a_flat[obs_flat] = mu_a_obs
    mu_a_2d = mu_a_flat.reshape(lat_n, lon_n)
    for _ in range(n_diff):
        mu_pad = np.pad(mu_a_2d, 1, mode='edge')
        laplacian = (mu_pad[:-2,1:-1]+mu_pad[2:,1:-1]+mu_pad[1:-1,:-2]+mu_pad[1:-1,2:]-4.0*mu_a_2d)
        update = np.zeros_like(mu_a_2d); update[~obs_mask_2d] = kappa*laplacian[~obs_mask_2d]
        mu_a_2d = mu_a_2d + 0.1*update
    return mu_a_2d, n_iter

print(f"\n{'Region':<20}{'Ocean%':>8}{'n_scored':>10}{'BG RMSE':>10}{'Recon RMSE':>12}{'Improvement':>13}")
print("-"*75)

results = []
for r0, c0, frac in sampled:
    r1, c1 = r0 + BOX, c0 + BOX
    rect_region = np.zeros_like(valid_mask)
    rect_region[r0:r1, c0:c1] = True
    test_mask = rect_region & valid_mask
    n_test = test_mask.sum()

    obs_mask = valid_mask.copy()
    obs_mask[test_mask] = False
    mu_b = field_prev.copy(); mu_b[np.isnan(mu_b)] = 0.0
    y_obs = np.where(obs_mask, field_curr, 0.0)

    mu_a, _ = cellnn_da_2d(mu_b, y_obs, obs_mask, alpha_A=1.0, wb=1.0, kappa=0.4, n_diff=50)

    l4_valid_in_mask = ~np.isnan(l4_field) & test_mask
    n_scored = l4_valid_in_mask.sum()
    if n_scored < 10:
        print(f"  rows {r0}-{r1}, cols {c0}-{c1}: only {n_scored} scoreable pixels, skipping")
        continue
    true_v = l4_field[l4_valid_in_mask]; an_v = mu_a[l4_valid_in_mask]; bg_v = mu_b[l4_valid_in_mask]
    rmse_an = np.sqrt(np.mean((true_v - an_v)**2))
    rmse_bg = np.sqrt(np.mean((true_v - bg_v)**2))
    imp = (rmse_bg - rmse_an) / rmse_bg * 100
    results.append((r0, c0, frac, n_scored, rmse_bg, rmse_an, imp))

    label = f"({r0},{c0})"
    print(f"{label:<20}{frac:>7.1%}{n_scored:>10}{rmse_bg:>10.4f}{rmse_an:>12.4f}{imp:>12.1f}%")

imps = [r[6] for r in results]
print("-"*75)
print(f"\nAcross {len(results)} blindly sampled regions: "
      f"mean improvement={np.mean(imps):.1f}%, "
      f"range=[{np.min(imps):.1f}%, {np.max(imps):.1f}%], "
      f"all positive: {all(i > 0 for i in imps)}")
print(f"\nCompare to Figure 11's region (190,0): 23.7% improvement (production n=50)")

np.savez('results/rect_mask_multiregion.npz',
         results=np.array(results, dtype=object))
print("\nSaved -> results/rect_mask_multiregion.npz")
print("Done.")
