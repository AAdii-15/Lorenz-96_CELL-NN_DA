"""
Fix: t=0 (2015-01-01) still has NaN gaps because
no background X(t-1) was available for the first step.

Fix: Use spatial diffusion (no temporal background)
to fill gaps in the very first time step only.
"""
import numpy as np

print("Fixing first time step (2015-01-01)...")

chl_recon = np.load('data/modis_chl/chl_reconstructed.npy')
chl_norm  = np.load('data/modis_chl/chl_norm.npy')

field = chl_norm[0].copy()
valid_mask = ~np.isnan(field)
print(f"Valid pixels in t=0: {valid_mask.sum()} / {valid_mask.size} "
      f"({100*valid_mask.mean():.1f}%)")

# Pure spatial diffusion fill (no prior background available)
field_fill = field.copy()
field_fill[~valid_mask] = 0.0

kappa  = 0.3
n_iter = 200

for i in range(n_iter):
    pad = np.pad(field_fill, 1, mode='edge')
    laplacian = (pad[:-2,1:-1] + pad[2:,1:-1] +
                 pad[1:-1,:-2] + pad[1:-1,2:] -
                 4.0*field_fill)
    update = np.zeros_like(field_fill)
    update[~valid_mask] = kappa * laplacian[~valid_mask]
    field_fill = field_fill + 0.1 * update

remaining_nan = np.isnan(field_fill).sum()
print(f"Remaining NaN after diffusion fill: {remaining_nan}")

chl_recon[0] = field_fill
np.save('data/modis_chl/chl_reconstructed.npy', chl_recon)

print(f"\nFinal check:")
print(f"  Total NaN in chl_reconstructed.npy: {np.isnan(chl_recon).sum()}")
print("Fixed and saved!")
