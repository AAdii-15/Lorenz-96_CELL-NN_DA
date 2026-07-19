"""
Three-way split for diffusion hyperparameter selection.
Reproduces the EXACT original 20% held-out mask (seed=42, same loop as
diffusion_sensitivity.py / cellnn_chl_da_proper.py), then splits it into
two disjoint 10%/10% halves per timestep using an independent seed (123).
"""
import numpy as np

chl_norm = np.load('data/modis_chl/chl_norm.npy')
N, LAT, LON = chl_norm.shape

# Step 1: reproduce the ORIGINAL 20% held-out mask exactly
np.random.seed(42)
held_out_20 = np.zeros((N, LAT, LON), dtype=bool)
eval_t = []
for t in range(1, N):
    field_curr = chl_norm[t]
    valid_mask = ~np.isnan(field_curr)
    if valid_mask.sum() < 500:
        continue
    valid_idx = np.where(valid_mask)
    n_valid = len(valid_idx[0])
    sel = np.random.choice(n_valid, int(0.2 * n_valid), replace=False)
    held_out_20[t, valid_idx[0][sel], valid_idx[1][sel]] = True
    eval_t.append(t)

print(f"Reproduced original 20% held-out mask. Timesteps: {len(eval_t)}")
print(f"Total held-out pixels: {held_out_20.sum()}")

# Step 2: split into two disjoint halves with an INDEPENDENT seed
np.random.seed(123)
val_mask  = np.zeros((N, LAT, LON), dtype=bool)
test_mask = np.zeros((N, LAT, LON), dtype=bool)

for t in eval_t:
    rows, cols = np.where(held_out_20[t])
    n_held = len(rows)
    perm = np.random.permutation(n_held)
    half = n_held // 2
    val_idx, test_idx = perm[:half], perm[half:]
    val_mask[t, rows[val_idx], cols[val_idx]] = True
    test_mask[t, rows[test_idx], cols[test_idx]] = True

overlap = (val_mask & test_mask).sum()
union_matches = np.array_equal(val_mask | test_mask, held_out_20)
print(f"\nValidation pixels: {val_mask.sum()}")
print(f"Test pixels:       {test_mask.sum()}")
print(f"Overlap (must be 0): {overlap}")
print(f"Union matches original 20% mask exactly: {union_matches}")

assert overlap == 0, "Validation and test masks overlap"
assert union_matches, "val+test union != original held-out mask"

np.savez('results/holdout_split_masks.npz',
         held_out_20=held_out_20, val_mask=val_mask,
         test_mask=test_mask, eval_t=np.array(eval_t))
print("\nSaved -> results/holdout_split_masks.npz")
print("Done.")
