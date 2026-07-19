# A Two-Condition Diagnostic for Relaxation-Based Data Assimilation

Research project testing when a relaxation-based data assimilation operator
(Cell-NN, extending Magno et al., JCS 2025) performs genuine assimilation
versus simple observation insertion. Tested on Lorenz-96 (genuine assimilation
regime) and MODIS-Aqua Chlorophyll-a reconstruction over the Bay of Bengal
(observation-insertion regime), validated against the independent Copernicus
L4 gap-free product. Target journal: Journal of Computational Science (JCS).

Research intern: Aditya Raj | Supervisor: Dr. Mahima Lakra (NITK)

---

## Status

Paper source (`final.tex`, `references.bib`) and all figures are in this repo,
current as of the latest Overleaf export. Peer-review-response work complete:
stronger EnKF baseline (RTPS), 3D-Var covariance tuning, multi-region
fixed-region test, validation/test leakage fix, all incorporated.

**Still open:**
- [ ] Exact seed lists for the 20-realization Lorenz-96 comparison and the
      5-seed gap-length robustness study are not yet enumerated anywhere
      (paper currently names these as "will be listed exhaustively in the
      public repository" — this repo is that promise, not yet fulfilled)
- [ ] Final compile check of `final.tex` in a clean Overleaf/local environment

---

## Project Structure
lorenz96_cellnn/
README.md                 this file
final.tex                 paper source (Overleaf export)
references.bib            bibliography
 requirements.txt
src/                       all analysis code, see Source Files Reference
data/
modis_chl/
     nc_files/               raw Aqua-MODIS L3 Chl-a, 230 files
reconstructed_nc/       Cell-NN reconstructed output, 230 files
chl_data.npy, chl_log.npy, chl_norm.npy, chl_reconstructed.npy
  chl_mean.npy, chl_std.npy, chl_lat.npy, chl_lon.npy, chl_dates.npy
    l4_chl/                   Copernicus L4 gap-free product (external
 validation reference, never used as model
input)
Figures/                    final paper figures
Supplementary/            supplementary figures
results/                    intermediate .npy/.npz/.png outputs
---

## Datasets

### 1. Aqua-MODIS L3 Chlorophyll-a (primary dataset)
- Source: NASA Ocean Color (oceandata.sci.gsfc.nasa.gov)
- Sensor: Aqua-MODIS | Product: Chlorophyll concentration, L3 Mapped
- Period: 8-day composite, 2015-01-01 to 2019-12-27 (230 composites)
- Region: Bay of Bengal, 80-100E, 5-22N
- Resolution: 4 km
- Mean valid coverage: 54.2%, driven mainly by SW monsoon cloud cover

### 2. Copernicus Marine Service L4 gap-free Chlorophyll-a (external validation)
- Multi-sensor, near-complete spatial coverage
- Used ONLY as an independent reference for validation, never supplied to
  the reconstruction pipeline as input
- Normalized with its OWN mean/std, independent of MODIS's normalization
- Aggregated to the same 8-day composite windows as MODIS for direct
  date-for-date comparison

---

## Preprocessing Pipeline

1. Raw Chl-a (mg/m3) -> log10(Chl-a), since Chl-a is approximately
   log-normally distributed
2. z-score normalize: (log10(x) - mean) / std, each product (MODIS, L4)
   normalized independently with its own mean/std

---

## Cell-NN DA Method

Core function: `src/da_cellnn.py` -> `cellnn_da_step()`. Same function, same
parameters (alpha_A=1.0, wb=1.0) used identically in both the Lorenz-96
experiments and the MODIS Chl-a reconstruction — no retraining between
domains.

Corrected relaxation equation (fixes a fixed-point bias in the original
Magno et al. 2025 formulation):

    d(mu_r)/d(tau) = -(mu_r - mu_obs) + alpha_A * clip(mu_obs - mu_r, -1, 1)

Fixed point: mu_r* = mu_obs exactly, for any alpha_A.

For Chl-a specifically (`src/cellnn_chl_da_proper.py`):
- Background mu_b(t) = X(t-1), the previous 8-day composite
- Observations y(t) = current composite's cloud-free pixels only
- Cloud gaps filled by Laplacian spatial diffusion (kappa=0.4, n=50
  iterations, selected on a validation subset disjoint from the reported
  test subset)

---

## Key Results Summary

### Lorenz-63 (reproduction of Magno et al. 2025)
3D-Var error = 0.360 | Cell-NN error = 0.373

### Lorenz-96 (novel extension, alpha_A=1.0)
3D-Var error = 0.344 | Cell-NN error = 0.310
R-score: 3D-Var=1.411, Cell-NN=1.245, 4DVarNet=0.594 (trained)
ODE-score: Cell-NN = 1.79e-13 (RK4 verification)
Across 20 realizations: Cell-NN beats 3D-Var in 14/20 (mean improvement
3.7%+/-4.2%, Wilcoxon p=0.0004, r=0.746)

### EnKF robustness under irregular observations
Fixed-inflation EnKF: continuous R-score=0.005, intermittent=23.965 (diverges)
RTPS adaptive-inflation EnKF: continuous R-score=0.0033-0.0058 (stable);
intermittent behaviour not reducible to a single interpretable number
(safety safeguard triggers on 99.3% of post-gap steps)
Cell-NN: 1.245 in both regimes (unchanged)

### 3D-Var covariance tuning (closes a reviewer-flagged fairness gap)
5x5 grid search over (sigma_b, sigma_r); best point found (0.7, 0.05)
reaches R-score=1.374, still worse than Cell-NN's 1.245 (9.4% gap)

### MODIS Chl-a internal validation (20% held-out pixels, test subset)
Background RMSE=0.7548 | OI=0.4909 (35.0%) | DINEOF=0.4125 (45.3%) |
Cell-NN pipeline=0.1330 (82.4%) | Monte Carlo=0.1195 (84.2%)
Cloud-shaped masking (realistic contiguous gaps): 20.8% improvement

### MODIS Chl-a external validation (vs independent L4)
Background RMSE=0.229, R^2=0.51, slope=0.64
Cell-NN RMSE=0.101, R^2=0.92, slope=1.04 (56.0% improvement)

### Attribution: what actually drives satellite reconstruction skill
2x2 ablation shows Cell-NN alone matches the raw background exactly
(0.7548) at every tested noise level (0-30%); Laplacian diffusion drives
all reconstruction skill, confirmed both numerically and via a fixed-region
test withholding an entire spatial block from the Cell-NN step (single
region: 23.7% improvement, growing to 32.5% at 1000 diffusion iterations;
5-region blind confirmation: mean 10.6%, range 0.8-21.5%, all positive)

### Computational cost
Cell-NN pipeline is 91x cheaper than Monte Carlo for satellite
reconstruction (11.9s vs 1080.0s, full 229-composite record); EnKF
dominates Lorenz-96 cost at every tested state dimension (K=10-160)

---

## Key Corrections Made During Development

Real bugs caught during review, documented so they are not repeated:

1. Background was initially a spatial mean (scalar), not real DA.
   FIXED: background = X(t-1), the previous composite.
2. External validation product was initially normalized using the primary
   sensor's own mean/std. FIXED: each product normalized independently.
3. R-score (Lorenz) was computed over the full trajectory including DA-OFF
   drift periods. FIXED: DA-ON periods only.
4. ODE-score was computed with Euler integration. FIXED: RK4.
5. First reconstructed timestep had residual NaNs (no background exists for
   the very first observation). FIXED with a one-off pure spatial-diffusion
   fill for that single timestep only.
6. Diffusion hyperparameters (kappa, n) were being tuned and reported on the
   same held-out pixels (leakage). FIXED: three-way split — held-out 20%
   excluded from all model input; disjoint 10% validation subset for
   hyperparameter selection; disjoint 10% test subset for all reported
   numbers.
7. EnKF's reported degradation used a single fixed-inflation configuration.
   FIXED: 14-configuration sensitivity sweep, plus a separate adaptive-
   inflation (RTPS) variant, both reported honestly including where RTPS's
   results were not cleanly interpretable.
8. 3D-Var's covariance parameters (sigma_b=0.5, sigma_r=0.1) were never
   tuned, unlike Cell-NN's alpha_A. FIXED: 5x5 grid search; Cell-NN still
   wins even against the best point found.
9. The fixed-region test used only one hand-selected region. FIXED: a
   blind, pre-registered 5-region sample added to confirm the pattern
   generalizes beyond the one region shown in the main figure.

---

## Source Files Reference (src/)

Lorenz systems and DA core:
  lorenz63.py, lorenz96.py            Lorenz-63 / Lorenz-96 systems
  da_cellnn.py                        Cell-NN DA core, cellnn_da_step()
  da_3dvar.py, lorenz96_da.py         3D-Var DA (Lorenz-63 / Lorenz-96)
  enkf.py                             EnKF, Gaspari-Cohn localization
  enkf_sensitivity.py, enkf_sensitivity_loc.py   14-configuration sweep
  enkf_rtps.py, enkf_rtps_comparison.py          RTPS adaptive inflation
  threedvar_sensitivity.py            3D-Var covariance grid search
  enkf_spread_plot.py                 EnKF spread diagnostic (Supp. Fig)
  template_sensitivity.py             Grid search over alpha_A, wb
  lyapunov_stability.py               Stability proof verification
  computational_scaling.py, plot_computational_scaling.py

Satellite Chl-a pipeline:
  preprocess_chl.py                   log10 + normalization pipeline
  cellnn_chl_da_proper.py             Chl-a Cell-NN DA, background=X(t-1)
  generate_split_masks.py             held-out/validation/test split
  diffusion_sensitivity_valsplit.py   diffusion kappa/n grid search
  cellnn_pipeline_testsplit.py, oi_baseline_testsplit.py,
  dineof_testsplit.py, montecarlo_testsplit.py    final method comparisons
  seasonal_chl_rmse_testsplit.py, noisy_modis_ablation_testsplit.py,
  stratified_comparison_testsplit.py, cellnn_hybrid_fallback_testsplit.py
  table6_bootstrap_ci.py, fix_seasonal_coverage.py
  satellite_cost_comparison.py        method wall-clock cost comparison
  l4_scatter_plot.py                  L4 predicted-vs-actual figure
  rect_mask_final.py                  fixed-region test (main figure)
  rect_mask_multiregion.py            blind 5-region confirmation
  study_region_context_map.py         study region figure
  multimethod_fig_final3.py           merged gap-filling comparison figure

---
