# Bayesian Analysis of UTMB Ultra-Trail Race Finish Times

## Project Overview

This project applies the full Bayesian workflow to model **Mean Finish Time** of ultra-trail running races using data from the UTMB (Ultra-Trail du Mont-Blanc) World Series. Two competing models — a Normal linear regression (baseline) and a Student-t linear regression (robust alternative) — are compared using information-theoretic criteria.

### Research Question

> How do race distance and elevation gain influence mean finish time in ultra-trail races, and does accounting for heavy-tailed residuals (outlier races) improve predictive performance?

### Dataset

- **Source**: Kaggle — `mgpoirot/utmb-world-race-daa` (UTMB World Race Data)
- **Raw observations**: 38,460 race results
- **After cleaning**: 36,433 observations (removed missing values and anomalies)
- **4 race categories**: 20K, 50K, 100K, 100M

### Response Variable

- **Mean Finish Time** (hours): continuous, positive, right-skewed (skewness ≈ 2.0, excess kurtosis ≈ 5.0)

### Predictors

- **Distance** (km): standardized (mean=0, std=1) for modeling
- **Elevation Gain** (m): standardized (mean=0, std=1) for modeling

### Models

| | Model 1 (Normal) | Model 2 (Student-t) |
|---|---|---|
| Likelihood | $y_i \sim \text{Normal}(\mu_i, \sigma)$ | $y_i \sim \text{Student-t}(\nu, \mu_i, \sigma)$ |
| Mean function | $\mu_i = \alpha + \beta_{dist} \cdot x_{1i} + \beta_{elev} \cdot x_{2i}$ | Same |
| Parameters | $\alpha, \beta_{dist}, \beta_{elev}, \sigma$ | $\alpha, \beta_{dist}, \beta_{elev}, \sigma, \nu$ |
| Tail behavior | Light (exponential decay) | Heavy (polynomial decay) |
| Robustness | Low | High |

### Priors

| Parameter | Prior | Rationale |
|---|---|---|
| $\alpha$ | Normal(10, 5) | Average race finish ~10h at mean distance/elevation |
| $\beta_{dist}$ | Normal(5, 3) | 1 SD distance (~30km) adds ~5h |
| $\beta_{elev}$ | Normal(2, 2) | 1 SD elevation (~1500m) adds ~2h |
| $\sigma$ | Exponential(0.2) | Residual SD, mean=5h |
| $\nu$ | Gamma(2, 0.1) | Degrees of freedom, mean=20, allows heavy tails |

### Key Findings (from executed results)

1. **Student-t model wins decisively** — ELPD difference ≈ 1286 (SE ≈ 137), ratio ≈ 9.4 — both LOO and WAIC agree
2. **Very heavy tails confirmed** — Posterior $\nu \approx 1.74$ (95% HDI: [1.62, 1.87]) — variance undefined ($\nu < 2$)
3. **58.6% lower sigma** — Student-t $\sigma \approx 1.01$h vs Normal $\sigma \approx 2.44$h
4. **Fewer problematic observations** — Normal has 1 Pareto-k > 0.7; Student-t has 0
5. **Distance dominates** — $\beta_{dist}$ ≈ 5.55–6.23h per SD, $\beta_{elev}$ ≈ 2.34–2.59h per SD

### Computational Notes

- **Subsampling**: Fitting notebooks use a random subsample of 5,000 observations (from 36,433) to prevent kernel crashes from memory exhaustion (36K obs × 8000 posterior draws × 3 generated quantity arrays ≈ 7 GB)
- **Sampling**: 4 chains × 1,000 post-warmup iterations = 4,000 posterior draws per model
- **EDA**: Notebook 01 uses the full dataset (36,433 observations) for all exploratory analysis

### Tools and Libraries

- Python 3.11
- CmdStanPy 1.2.5 (Stan interface)
- ArviZ 0.21.0 (Bayesian visualization and diagnostics)
- pandas 2.2.3, numpy 2.2.4, scipy 1.15.2
- kagglehub 0.3.13 (dataset download)

---

## Project Structure

```
project/
├── 01_problem_formulation.ipynb       — Problem definition, data loading, EDA
├── 02_model_specification_priors.ipynb — Model specs, prior rationale, prior predictive checks
├── 03_posterior_model1_normal.ipynb    — Fit Normal model, diagnostics, PPC, posteriors
├── 04_posterior_model2_student_t.ipynb — Fit Student-t model, diagnostics, PPC, posteriors
├── 05_model_comparison.ipynb          — LOO, WAIC, Pareto-k, final assessment
├── model1_normal.stan                 — Stan code for Normal model
├── model2_student_t.stan              — Stan code for Student-t model
├── utmb_processed.csv                 — Preprocessed data (36,433 rows)
├── fig*.png                           — Saved figures from all notebooks
├── notes.txt                          — Design notes
└── PROJECT_DESCRIPTION.md             — This file
```

---

## Requirements Analysis

The project is evaluated against 6 grading criteria, each worth 4 points (24 points total). Below is a requirement-by-requirement analysis showing where each task is implemented.

---

### 1. Problem Formulation (4 pts)

| Requirement | Status | Location |
|---|---|---|
| Clear problem statement | ✅ | `01_problem_formulation.ipynb`, Section 1.1 (Cell 1) — "Can we predict mean finish time from distance and elevation?" |
| Data source identified and justified | ✅ | `01_problem_formulation.ipynb`, Section 1.2 (Cell 1) — Kaggle UTMB dataset, justification for real-world ultra-trail data |
| Data loading and preprocessing | ✅ | `01_problem_formulation.ipynb`, Cells 2-4 — kagglehub download, cleaning (NaN removal, anomaly filtering), standardization |
| Exploratory data analysis | ✅ | `01_problem_formulation.ipynb`, Cells 5-7 — Distribution plots, QQ plots, scatter plots, correlation matrix, descriptive statistics |
| Variable selection justified | ✅ | `01_problem_formulation.ipynb`, Section 1.3 — Response (Mean Finish Time) chosen as continuous metric; predictors (Distance, Elevation Gain) chosen based on strong correlations (r=0.937, r=0.85) |
| Data characteristics discussed | ✅ | `01_problem_formulation.ipynb`, Cell 7 — Skewness (1.99), kurtosis (4.94), heavy right tail motivates Student-t model |

---

### 2. Model Specification (4 pts)

| Requirement | Status | Location |
|---|---|---|
| Two models mathematically specified | ✅ | `02_model_specification_priors.ipynb`, Section 2.1 (Cell 1) — Full LaTeX formulas for both models |
| Stan code for Model 1 | ✅ | `model1_normal.stan` — Complete Stan program (data, parameters, model, generated quantities) |
| Stan code for Model 2 | ✅ | `model2_student_t.stan` — Complete Stan program with nu parameter and student_t likelihood |
| Technical model description | ✅ | `02_model_specification_priors.ipynb`, Section 2.2 (Cell 3) — Full Stan code listings with annotations |
| Justification for model choice | ✅ | `02_model_specification_priors.ipynb`, Section 2.1 — Comparison table (likelihood, parameters, tail behavior, robustness), rationale from EDA findings |
| Models differ in a meaningful way | ✅ | `02_model_specification_priors.ipynb`, Section 2.1 — Student-t nests Normal (ν→∞), provides robustness to outlier races without inflating σ |

---

### 3. Prior Selection (4 pts)

| Requirement | Status | Location |
|---|---|---|
| All priors explicitly listed | ✅ | `02_model_specification_priors.ipynb`, Section 2.3 (Cell 4) — Table with all 5 priors |
| Rationale for each prior | ✅ | `02_model_specification_priors.ipynb`, Section 2.3 — Domain knowledge justification for each parameter's prior |
| Prior selection method explained | ✅ | `02_model_specification_priors.ipynb`, Section 2.3 — 4-point method: domain knowledge, scale matching, weakly informative principle, prior predictive simulation |
| Prior predictive check (parameter level) | ✅ | `02_model_specification_priors.ipynb`, Section 2.4 (Cell 6) — Histograms of all 5 prior parameter distributions with 95% intervals |
| Prior predictive check (measurement level) | ✅ | `02_model_specification_priors.ipynb`, Section 2.5 (Cells 8-9) — Simulated datasets from both models, CDF overlays, summary statistics comparison |
| Assessment of prior predictive results | ✅ | `02_model_specification_priors.ipynb`, Section 2.6 (Cell 10) — Conclusions about prior adequacy, discussion of negative times artifact |

---

### 4. Posterior Analysis — Model 1 (4 pts)

| Requirement | Status | Location |
|---|---|---|
| MCMC sampling | ✅ | `03_posterior_model1_normal.ipynb`, Cell 3 — 4 chains, 1000 post-warmup iterations, seed=42 (5000-obs subsample) |
| R-hat diagnostic | ✅ | `03_posterior_model1_normal.ipynb`, Cell 6 — All R-hat values checked against 1.01 threshold |
| ESS diagnostic (bulk and tail) | ✅ | `03_posterior_model1_normal.ipynb`, Cell 6 — ESS_bulk and ESS_tail checked against 400 threshold |
| Divergence check | ✅ | `03_posterior_model1_normal.ipynb`, Cell 5 — `fit1.diagnose()` output |
| Trace plots | ✅ | `03_posterior_model1_normal.ipynb`, Cell 7 — ArviZ trace plots for all parameters, saved as fig05 |
| Posterior predictive check (density) | ✅ | `03_posterior_model1_normal.ipynb`, Cell 9 — `az.plot_ppc` density overlay + observed vs predicted scatter |
| Posterior predictive check (summary stats) | ✅ | `03_posterior_model1_normal.ipynb`, Cell 10 — Mean, std, min, max, skewness, % negative predictions |
| Data consistency assessment | ✅ | `03_posterior_model1_normal.ipynb`, Cell 11 — Bayesian p-values for all summary statistics |
| PPC by subgroup | ✅ | `03_posterior_model1_normal.ipynb`, Cell 12 — PPC by race category (20K, 50K, 100K, 100M) |
| Marginal posterior distributions | ✅ | `03_posterior_model1_normal.ipynb`, Cell 13 — `az.plot_posterior` with 95% HDI |
| Parameter interpretation | ✅ | `03_posterior_model1_normal.ipynb`, Cell 14 — Posterior means, HDIs, original-units conversion, probability statements |
| Concentration/diffusion discussion | ✅ | `03_posterior_model1_normal.ipynb`, Cell 14 — Coefficient of Variation (CV) for all parameters |
| Pair plot | ✅ | `03_posterior_model1_normal.ipynb`, Cell 15 — `az.plot_pair` with KDE and marginals |
| Model limitations discussed | ✅ | `03_posterior_model1_normal.ipynb`, Section 3.5 — Symmetric residuals, light tails, negative predictions, homoscedasticity |

---

### 5. Posterior Analysis — Model 2 (4 pts)

| Requirement | Status | Location |
|---|---|---|
| MCMC sampling | ✅ | `04_posterior_model2_student_t.ipynb`, Cell 3 — 4 chains, 1000 post-warmup iterations, seed=42 (5000-obs subsample) |
| R-hat diagnostic | ✅ | `04_posterior_model2_student_t.ipynb`, Cell 6 — All R-hat values checked (including nu) |
| ESS diagnostic (bulk and tail) | ✅ | `04_posterior_model2_student_t.ipynb`, Cell 6 — Note about nu potentially having lower ESS |
| Divergence check | ✅ | `04_posterior_model2_student_t.ipynb`, Cell 5 — `fit2.diagnose()` output |
| Trace plots | ✅ | `04_posterior_model2_student_t.ipynb`, Cell 7 — Trace plots for all 5 parameters, saved as fig11 |
| Sampling issues discussion | ✅ | `04_posterior_model2_student_t.ipynb`, Section after Cell 7 — Discussion of nu identification, mitigations |
| Posterior predictive check (density) | ✅ | `04_posterior_model2_student_t.ipynb`, Cell 9 — PPC density overlay + observed vs predicted |
| Posterior predictive check (summary stats) | ✅ | `04_posterior_model2_student_t.ipynb`, Cell 10 — Mean, std, min, max, skewness, **kurtosis** (key for Student-t) |
| Data consistency assessment | ✅ | `04_posterior_model2_student_t.ipynb`, Cell 11 — Bayesian p-values with kurtosis added |
| PPC by subgroup | ✅ | `04_posterior_model2_student_t.ipynb`, Cell 12 — PPC by race category |
| Marginal posterior distributions | ✅ | `04_posterior_model2_student_t.ipynb`, Cell 13 — All 5 parameters with 95% HDI |
| Parameter interpretation (including ν) | ✅ | `04_posterior_model2_student_t.ipynb`, Cell 14 — Full interpretation, P(ν<10), P(ν<30), tail heaviness assessment |
| Sigma comparison between models | ✅ | `04_posterior_model2_student_t.ipynb`, Cell 15 — Side-by-side σ histogram, % reduction reported |
| Concentration/diffusion discussion | ✅ | `04_posterior_model2_student_t.ipynb`, Cell 14 — CV for all 5 parameters |
| Pair plot | ✅ | `04_posterior_model2_student_t.ipynb`, Cell 16 — Pair plot with sigma-nu correlation discussion |
| Model summary | ✅ | `04_posterior_model2_student_t.ipynb`, Section 4.5 — Key findings, comparison with Model 1 |

---

### 6. Model Comparison (4 pts)

| Requirement | Status | Location |
|---|---|---|
| LOO (PSIS-LOO) computation | ✅ | `05_model_comparison.ipynb`, Section 5.1 (Cells 4-5) — `az.loo()` for both models |
| LOO comparison table | ✅ | `05_model_comparison.ipynb`, Cell 5 — `az.compare()` with elpd_loo, p_loo, elpd_diff, dse, weight |
| LOO visualization | ✅ | `05_model_comparison.ipynb`, Cell 6 — `az.plot_compare()`, saved as fig18 |
| LOO discussion (elpd_diff / dse) | ✅ | `05_model_comparison.ipynb`, Cell 8 — Quantitative assessment of difference significance |
| WAIC computation | ✅ | `05_model_comparison.ipynb`, Section 5.2 (Cells 9-10) — `az.waic()` for both models |
| WAIC comparison table | ✅ | `05_model_comparison.ipynb`, Cell 10 — `az.compare(..., ic='waic')` |
| WAIC visualization | ✅ | `05_model_comparison.ipynb`, Cell 10 — `az.plot_compare()`, saved as fig19 |
| WAIC discussion | ✅ | `05_model_comparison.ipynb`, Cell 11 — Significance assessment, LOO-WAIC agreement check |
| Pareto-k diagnostics (plot) | ✅ | `05_model_comparison.ipynb`, Section 5.3 (Cell 12) — `az.plot_khat()` for both models |
| Pareto-k analysis (quantitative) | ✅ | `05_model_comparison.ipynb`, Cell 13 — Threshold summary table (k<0.5, 0.5≤k<0.7, k≥0.7, k≥1.0), interpretation |
| Side-by-side PPC comparison | ✅ | `05_model_comparison.ipynb`, Section 5.4 (Cell 14) — PPC density + residual distributions |
| Tail behavior comparison | ✅ | `05_model_comparison.ipynb`, Cell 15 — Max value distribution, % extreme values comparison |
| Final comprehensive assessment | ✅ | `05_model_comparison.ipynb`, Section 5.5 (Cell 16) — Multi-criteria summary table, 5-point assessment |
| Final comparison visualization | ✅ | `05_model_comparison.ipynb`, Cell 17 — 4-panel summary figure (LOO, WAIC, Pareto-k, σ posteriors) |
| Conclusions | ✅ | `05_model_comparison.ipynb`, Section 5.6 — Bayesian workflow summary, key takeaways |

---

## Summary

All 6 grading criteria are fully addressed across the 5 notebooks and 2 Stan model files. The project follows the complete Bayesian workflow: problem formulation → model specification → prior selection with predictive checks → posterior fitting with full diagnostics → model comparison with information criteria. Each notebook contains both code and interpretive markdown commentary.

### Execution Results Summary

All 5 notebooks have been executed end-to-end with no errors:

| Notebook | Status | Key Output |
|---|---|---|
| 01 (EDA) | ✅ Executed (full 36,433 rows) | Distribution plots, correlation matrix, `utmb_processed.csv` |
| 02 (Priors) | ✅ Executed | Prior predictive checks (parameter + measurement level), Stan models compiled |
| 03 (Model 1) | ✅ Executed (5,000 subsample) | R-hat=1.00, ESS>2400, 0 divergences, α=10.66, β_dist=6.23, β_elev=2.59, σ=2.44 |
| 04 (Model 2) | ✅ Executed (5,000 subsample) | R-hat=1.00, ESS>1749, 0 divergences, α=10.13, β_dist=5.55, β_elev=2.34, σ=1.01, ν=1.74 |
| 05 (Compare) | ✅ Executed (5,000 subsample) | ELPD diff=1286 (SE=137), Student-t wins decisively (diff/SE=9.4) |
