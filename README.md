# Cell-NN Data Assimilation: Lorenz Systems to Real-World Satellite Chlorophyll-a

Research project extending Cell-NN Data Assimilation (Magno et al., JCS 2025)
from Lorenz-63 to Lorenz-96, and then to real MODIS-Aqua Chlorophyll-a
reconstruction over the Bay of Bengal, validated against the independent
SNPP-VIIRS sensor.

Research intern: Aditya Raj | Supervisor: Dr. Mahima Lakra (NITK)

---

## Status (as of 20 June 2026)

- [x] Lorenz-63: Cell-NN DA reproduced (matches Magno et al. 2025)
- [x] Lorenz-96: Cell-NN templates derived, fixed-point bias corrected
- [x] Lyapunov stability proof (global, both L63 and L96)
- [x] Benchmarked vs 3D-Var, EnKF, 4DVarNet (R-score, ODE-score)
- [x] EnKF divergence under intermittent observations (novel finding)
- [x] MODIS Chl-a (Bay of Bengal, 2015-2019) downloaded and preprocessed
- [x] Cell-NN DA applied to MODIS Chl-a, background = X(t-1)
- [x] Internal mask-and-recover validation (76.3% improvement)
- [x] VIIRS external validation (60.5% improvement, independently normalized)
- [x] Cell-NN reconstruction saved as 230 .nc files
- [ ] OC-CCI / Monte-Carlo broad-scale validation (pending, see Open Items)
- [ ] Paper: Literature Review/Background done, Methodology done,
      Results done except OC-CCI section, Discussion/Conclusion drafted,
      Introduction not yet written (per Mam's instruction, written last)

---

## Project Structure

lorenz96_cellnn/
  README.md                      this file
  requirements.txt
  src/                           all analysis code, see Source Files Reference
  data/
    modis_chl/
      nc_files/                  raw Aqua-MODIS L3 Chl-a, 230 files
      reconstructed_nc/          Cell-NN reconstructed output, 230 files
      chl_data.npy               raw Chl-a array (230, 408, 480), mg/m3
      chl_log.npy                log10(Chl-a)
      chl_norm.npy               z-score normalized log10(Chl-a), has NaN gaps
      chl_reconstructed.npy      Cell-NN-filled normalized array, no NaN
      chl_mean.npy, chl_std.npy  MODIS normalization stats
      chl_lat.npy, chl_lon.npy, chl_dates.npy
    viirs_chl/
      nc_files/                  raw SNPP-VIIRS L3 Chl-a, 230 files
    occci_chl/
      OC_CCI_interp_2015.nc ... 2019.nc   monthly, 0.25deg, from Mam via Drive
  results/                       all .png plots and .npy score files

---

## Datasets

### 1. Aqua-MODIS L3 Chlorophyll-a (primary dataset)
- Source: NASA Ocean Color (oceandata.sci.gsfc.nasa.gov), Earthdata login required
- Sensor: Aqua-MODIS | Product: Chlorophyll concentration, L3 Mapped
- Period: 8-day composite | Time range: 2015-01-01 to 2019-12-27 (230 files)
- Region: Bay of Bengal, 80-100E, 5-22N (per Mam's instruction)
- Resolution: 4 km, grid shape (lat=408, lon=480)
- Missing data: 59.3% average (cloud cover), range 27%-95.2%
- Download method: curl + NASA Earthdata .netrc auth, bulk order via
  oceandata.sci.gsfc.nasa.gov/l3/order/, extracted from TAR archive

### 2. SNPP-VIIRS L3 Chlorophyll-a (independent validation dataset)
- Same source, region, period, compositing window, resolution as MODIS above
- Used ONLY for external validation, never as an input to Cell-NN
- Normalized with ITS OWN mean/std, NOT MODIS's - this was a corrected bug,
  see Key Corrections below

### 3. ESA OC-CCI (in progress, broad-scale validation)
- Source: shared by Mam via Google Drive (interpolated/gap-filled version)
- Files: OC_CCI_interp_2015.nc through OC_CCI_interp_2019.nc
- Resolution: 0.25 deg (~25km), MONTHLY (12 timesteps/file)
- Variable name: chloro_interp
- NOTE: 6x coarser than MODIS and monthly vs 8-day, resolution-matching
  protocol NOT yet finalized with Mam. Do not use for any reported result yet.

---

## Preprocessing Pipeline (MODIS and VIIRS, identical steps)

1. Raw Chl-a (mg/m3) -> log10(Chl-a), since Chl-a is lognormally distributed
2. z-score normalize: (log10(x) - mean) / std
   IMPORTANT: each sensor uses its OWN mean/std, computed independently
   - MODIS: mean=-0.6540, std=0.3361
   - VIIRS: mean=-0.6423, std=0.3293
   (difference of only 0.012 confirms cross-sensor agreement)

---

## Cell-NN DA Method (applied identically to Lorenz-96 and to Chl-a)

Core function: src/da_cellnn.py -> cellnn_da_step()
Same function, same parameters (alpha_A=1.0, wb=1.0) used in BOTH the
Lorenz-96 experiments and the MODIS Chl-a reconstruction. This is the
central claim of the paper: one operator, no retraining, two domains.

Corrected relaxation equation (fixes a fixed-point bias in the original
Magno et al. 2025 formulation):
    d(mu_r)/d(tau) = -(mu_r - mu_obs) + alpha_A * clip(mu_obs - mu_r, -1, 1)
Fixed point: mu_r* = mu_obs exactly, for any alpha_A.

For Chl-a specifically (src/cellnn_chl_da_proper.py):
- Background mu_b(t) = X(t-1), the PREVIOUS 8-day composite (NOT a spatial
  mean, that was an earlier incorrect attempt, see Key Corrections)
- Observations y(t) = current composite's cloud-free pixels only
- Flatten 2D field, apply cellnn_da_step() to observed pixels only,
  reshape back to 2D, fill remaining cloud gaps via spatial diffusion
  (Laplacian smoothing, kappa=0.3)

---

## Key Results Summary

### Lorenz-63 (reproduction)
3D-Var error = 0.3603 (paper ~0.36) | Cell-NN error = 0.3726 (paper ~0.37)

### Lorenz-96 (novel, optimized params: alpha_A=1.0, wb=1.0)
3D-Var error = 0.3435 | Cell-NN error = 0.3103 (9.7% better)
R-score: 3D-Var=1.4106, Cell-NN=1.2452, 4DVarNet=0.5944 (trained)
ODE-score: Cell-NN = 1.79e-13 (RK4), about 9 orders better than Fablet's <1e-4

### EnKF robustness (key novel finding)
EnKF continuous: R-score=0.0051 (stable)
EnKF intermittent (ON/OFF): R-score=23.9647 (DIVERGES)
Cell-NN intermittent: R-score=1.2452 (stable, barely changed)

### MODIS Chl-a internal validation (mask 20% of valid pixels, recover)
Background(t-1) RMSE=0.7549 | Cell-NN RMSE=0.1785 | Improvement=76.3%
Consistent 2016-2019: 74-79%

### MODIS Chl-a external validation (vs independent VIIRS)
Background vs VIIRS RMSE=0.2394 | Cell-NN vs VIIRS RMSE=0.0946
Improvement=60.5%, consistent 2016-2019: 59-68%

---

## Key Corrections Made During Development

These were real bugs Mam caught during review, documented here so they
are not repeated, and because the reasoning behind each fix is relevant
to how rigorously the Methodology section should be written:

1. Background was initially a spatial mean (scalar), not real DA, just
   interpolation. FIXED: background = X(t-1), the previous composite.
2. VIIRS was initially normalized using MODIS's mean/std. FIXED: each
   sensor normalized independently with its own mean/std.
3. VIIRS validation initially compared Cell-NN-filled values at MODIS's
   natural gap locations against VIIRS, but VIIRS is also often missing
   at the same locations (correlated cloud cover), biasing RMSE upward.
   FIXED: explicitly mask 20% of MODIS's VALID pixels (independent of
   VIIRS's own gaps) and compare only there.
4. R-score (Lorenz) was computed over the full trajectory including DA-OFF
   drift periods. FIXED: DA-ON periods only, matching Fablet et al. 2021.
5. ODE-score was computed with Euler integration (gave 4.26e-07). FIXED:
   RK4, matching Fablet et al. 2021 (gives 1.79e-13).
6. Noise-sensitivity experiment had a bug: noise_level=noise/2 instead of
   noise, silently halving the intended noise level. FIXED.
7. First reconstructed timestep (2015-01-01) had residual NaNs because no
   background X(t-1) exists for the very first observation. FIXED with a
   one-off pure spatial-diffusion fill for that single timestep only.

---

## Open Items / Not Yet Done

1. OC-CCI / Monte-Carlo broad-scale validation. Mam's PhD student Gokul
   is running an independent Monte-Carlo gap-filling computation on the
   same region for comparison. Resolution-matching protocol (coarsen
   MODIS to OC-CCI's 0.25deg grid, NOT upsample OC-CCI) proposed to Mam,
   not yet confirmed.
2. Tasks requested by Mam, not yet done:
   - Single overall RMSE, full 2015-2019, log10 scale, ignore landmask
   - Time-series of spatial-mean-per-timestep, single overlay panel
     (Cell-NN reconstructed vs L4/ground-truth), to check extreme-value
     (bloom peak) recovery specifically
   - Spatial map snapshot comparison, 1 January 2016, Cell-NN-reconstructed
     vs L4/ground-truth sensor, same date
3. Reading: 3-4 relevant papers before finalizing writing, specifically
   Elodie Martinez papers (not yet identified/sourced).
4. Paper: Introduction section (Mam wants this written last).

---

## Source Files Reference (src/)

lorenz63.py               Lorenz-63 system
lorenz96.py                Lorenz-96 system
cellnn.py                  Cell-NN as time integrator (template check only, not DA)
da_3dvar.py                3D-Var DA
da_cellnn.py                Cell-NN DA core, cellnn_da_step(), used everywhere
enkf.py                     EnKF with Gaspari-Cohn localization
enkf_reproduce.py            EnKF Always-ON vs ON/OFF comparison
lorenz96_da.py               Lorenz-96 DA experiment driver
lorenz96_da_stable.py         Final noise-level analysis
compute_scores_maps.py        R-score (DA-ON only) + ODE-score (RK4)
template_sensitivity.py       Grid search over alpha_A, wb
lyapunov_stability.py         Stability proof verification
train_4dvarnet.py             4DVarNet training (PyTorch)
save_true_state.py            Saves L63/L96 true states
visualize_chl.py              Chl-a spatial/seasonal/time-series plots
preprocess_chl.py             Chl-a log10 + normalization pipeline
cellnn_chl_da_proper.py       Chl-a Cell-NN DA, background=X(t-1)
cellnn_reconstruct_save.py    Saves full reconstructed Chl-a array, no NaN
plot_paper_figures.py         Paper-style seasonal/time-series plots
validate_viirs_fixed2.py      FINAL VIIRS validation, correct normalization
save_reconstruction_nc.py     Saves reconstruction as 230 .nc files
fix_first_timestep.py         One-off NaN fix for t=0

