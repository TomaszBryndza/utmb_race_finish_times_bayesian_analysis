# Bayesian Analysis of UTMB Ultra-Trail Race Data

This repository contains a complete Bayesian workflow for analysing UTMB World Series race-level data. The project models three related phenomena:

1. **Mean Finish Time** — the average race finish time.
2. **Winning Time** — the finish time achieved by the race winner.
3. **Female Participation** — the number and share of female participants.

The analysis is implemented in **Python**, **Jupyter Notebook**, **Stan**, **CmdStanPy** and **ArviZ**. The workflow follows the full Bayesian modelling cycle: problem formulation, exploratory data analysis, prior design, prior predictive checks, posterior sampling, sampling diagnostics, posterior predictive checks and model comparison using **PSIS-LOO** and **WAIC**.

The main modelling principle is:

> Use a likelihood that matches the mathematical nature of the response variable.

Positive race times are modelled on the logarithmic scale, while female participation is modelled as a bounded count conditioned on the total number of participants.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Research Questions](#research-questions)
- [Current Report Notebooks](#current-report-notebooks)
- [Repository Structure](#repository-structure)
- [Feature Engineering](#feature-engineering)
- [Model Specifications](#model-specifications)
- [Prior Choices](#prior-choices)
- [Model Comparison Strategy](#model-comparison-strategy)
- [Main Results](#main-results)
- [Generated Outputs](#generated-outputs)
- [How to Run](#how-to-run)
- [Installation](#installation)
- [Reproducibility Notes](#reproducibility-notes)
- [Troubleshooting Notes](#troubleshooting-notes)
- [Limitations and Future Work](#limitations-and-future-work)
- [Final Project Conclusion](#final-project-conclusion)

---

## Project Overview

Ultra-trail race performance depends on multiple interacting factors: distance, elevation gain, route steepness, altitude, race category, geography, race year and unobserved race-specific conditions. Races with similar nominal distance and elevation can still differ because of weather, trail technicality, organization, participant profile and regional differences.

A Bayesian approach is useful because it provides:

- full posterior uncertainty for model parameters;
- explicit prior assumptions;
- posterior predictive distributions instead of point predictions only;
- robust alternatives for outliers and heavy-tailed residuals;
- principled predictive comparison through PSIS-LOO and WAIC.

The project compares simple baseline models with more robust likelihoods:

- **Normal vs Student-t** for log-time models;
- **Binomial vs Beta-Binomial** for female participation counts.

---

## Dataset

The project uses race-level data from the UTMB World Series dataset:

```text
mgpoirot/utmb-world-race-daa
```

The raw file expected by the notebooks is:

```text
utmb-race-data-sheet.csv
```

Some notebooks can also use already processed files:

```text
utmb_processed.csv
utmb_female_participation_processed.csv
```

The analysis focuses on the four main UTMB race categories:

```text
20K, 50K, 100K, 100M
```

Important variables used in the project:

| Variable | Meaning | Used in |
|---|---|---|
| `Race Category` | UTMB category: `20K`, `50K`, `100K`, `100M` | EDA, grouped PPC |
| `Distance` | Race distance in kilometres | all models |
| `Elevation Gain` | Total positive elevation gain in metres | all models |
| `Elevation` | Altitude above sea level in metres | time and participation models |
| `Mean Finish Time` | Average finish time in hours | mean-time model |
| `Winning Time` | Winner's finish time in hours | winning-time model |
| `N Participants` | Total number of race participants | female participation model |
| `N Women` | Number of female participants | female participation model |
| `Year` | Race year | female participation model |
| `Longitude`, `Latitude` | Race/event coordinates | female participation model |
| `Country` | Race country | EDA and possible future hierarchical models |

The analysis is **race-level**, not runner-level. Each row is an aggregated race observation.

---

## Research Questions

The project addresses the following questions:

1. How do distance, elevation gain, steepness and altitude influence expected race finish times?
2. Is log-scale modelling appropriate for positive and right-skewed race times?
3. Does a Student-t likelihood improve predictive performance over a Normal likelihood for time models?
4. How do distance, elevation, steepness, altitude, geography and year affect female participation?
5. Is ordinary Binomial variation sufficient for female participation counts, or is overdispersion present?
6. Which model should be preferred for each modelling task according to PSIS-LOO, WAIC and posterior predictive checks?

---

## Current Report Notebooks

The current cleaned report notebooks are:

| Report notebook | Topic | Compared models |
|---|---|---|
| `final_bayesian_utmb_report_katex_fixed.ipynb` | Mean finish time | Model 1 Normal vs Model 2 Student-t |
| `final_bayesian_utmb_winning_time_report_text_fixed.ipynb` | Winning time | Model 3 Normal vs Model 4 Student-t |
| `final_bayesian_utmb_female_participation_report.ipynb` | Female participation | Model 7 Binomial vs Model 8 Beta-Binomial |

These final notebooks are report-oriented. They avoid broken, stale or misleading exploratory plots from earlier notebooks and include cleaner EDA, model formulation, priors, prior predictive checks, posterior diagnostics, posterior predictive checks and model comparison.

The older exploratory notebooks are still useful as development history:

| Notebook | Role |
|---|---|
| `01_problem_formulation.ipynb` | Initial data loading, cleaning and problem formulation |
| `02_model_specification_priors.ipynb` | Initial model/prior work and prior predictive experiments |
| `03_posterior_model1_normal.ipynb` | Posterior analysis for mean-time Normal model |
| `04_posterior_model2_student_t.ipynb` | Posterior analysis for mean-time Student-t model |
| `06_winning_time_log_modeling.ipynb` | Development notebook for winning-time log-scale models |
| `07_female_participation_modeling.ipynb` | Development notebook for female participation models |

---

## Repository Structure

Recommended project structure:

```text
.
├── README.md
├── utmb-race-data-sheet.csv
├── utmb_processed.csv
├── utmb_female_participation_processed.csv
│
├── final_bayesian_utmb_report_katex_fixed.ipynb
├── final_bayesian_utmb_winning_time_report_text_fixed.ipynb
├── final_bayesian_utmb_female_participation_report.ipynb
│
├── 01_problem_formulation.ipynb
├── 02_model_specification_priors.ipynb
├── 03_posterior_model1_normal.ipynb
├── 04_posterior_model2_student_t.ipynb
├── 06_winning_time_log_modeling.ipynb
├── 07_female_participation_modeling.ipynb
│
├── model1_normal.stan
├── model2_student_t.stan
├── model3_winning_log_normal.stan
├── model4_winning_log_student_t.stan
├── model7_female_binomial.stan
├── model8_female_beta_binomial.stan
│
├── idata_model1_normal.nc
├── idata_model2_student_t.nc
├── idata_winning_log_normal.nc
├── idata_winning_log_student_t.nc
├── idata_female_binomial.nc
├── idata_female_beta_binomial.nc
│
├── report_figures/
├── report_figures_winning_time/
└── report_figures_female_participation/
```

The `idata_*.nc` files are optional caches. If they are absent, the notebooks fit the Stan models again.

---

## Feature Engineering

Continuous predictors are transformed and standardized before modelling. Standardization makes priors easier to define and makes regression coefficients comparable across predictors.

### Time models

Used for **Mean Finish Time** and **Winning Time**:

| Predictor | Construction | Purpose |
|---|---|---|
| `distance_log_std` | standardized `log(Distance)` | captures multiplicative distance effects |
| `elevation_log_std` | standardized `log1p(Elevation Gain)` | handles skewed positive elevation gain |
| `steepness_std` | standardized `log1p(Elevation Gain / Distance)` | separates route steepness from total elevation |
| `altitude_std` | standardized `log1p(Elevation)` | captures altitude-related difficulty |

For time models, the response is log-transformed:

```text
log_time = log(Mean Finish Time)
log_winning_time = log(Winning Time)
```

Back-transformation is performed with:

```text
time = exp(log_time)
winning_time = exp(log_winning_time)
```

This guarantees positive predictions on the original hour scale.

### Female participation model

The female participation model uses the same course-difficulty predictors and adds geography and year:

| Predictor | Construction | Purpose |
|---|---|---|
| `distance_log_std` | standardized `log(Distance)` | course length effect |
| `elevation_gain_log_std` | standardized `log1p(Elevation Gain)` | elevation gain effect |
| `steepness_std` | standardized `log1p(Elevation Gain / Distance)` | steepness effect |
| `altitude_std` | standardized `log1p(Elevation)` | altitude effect |
| `longitude_std` | standardized longitude | broad geographic differences |
| `latitude_std` | standardized latitude | broad geographic differences |
| `year_std` | standardized year | temporal trend |

The response is a bounded count:

```text
0 <= n_women <= n_participants
female_share = n_women / n_participants
```

---

## Model Specifications

### Mean Finish Time: Model 1 and Model 2

Target:

```text
y_i = log(Mean Finish Time_i)
```

Linear predictor:

```text
mu_i = alpha
     + beta_dist  * distance_log_std_i
     + beta_elev  * elevation_log_std_i
     + beta_steep * steepness_std_i
     + beta_alt   * altitude_std_i
```

Model 1 — Normal likelihood:

```text
y_i ~ Normal(mu_i, sigma)
```

Model 2 — Student-t likelihood:

```text
y_i ~ Student_t(nu, mu_i, sigma)
nu = 2 + nu_minus_two
```

Generated quantities include:

```text
mu
time_mu = exp(mu)
log_time_rep
time_rep = exp(log_time_rep)
log_lik
```

---

### Winning Time: Model 3 and Model 4

Target:

```text
y_i = log(Winning Time_i)
```

Linear predictor:

```text
mu_i = alpha
     + beta_dist  * distance_log_std_i
     + beta_elev  * elevation_log_std_i
     + beta_steep * steepness_std_i
     + beta_alt   * altitude_std_i
```

Model 3 — Normal likelihood:

```text
y_i ~ Normal(mu_i, sigma)
```

Model 4 — Student-t likelihood:

```text
y_i ~ Student_t(nu, mu_i, sigma)
nu = 2 + nu_minus_two
```

Generated quantities include:

```text
mu
winning_time_mu = exp(mu)
log_winning_time_rep
winning_time_rep = exp(log_winning_time_rep)
log_lik
```

Because the model is fitted on the log scale, exponentiating a coefficient gives a multiplicative interpretation:

```text
exp(beta_j)
```

Equivalently:

```text
(exp(beta_j) - 1) * 100%
```

is the approximate percentage change in expected time for a one-standard-deviation increase in a predictor, holding other predictors fixed.

---

### Female Participation: Model 7 and Model 8

Let:

```text
Y_i = n_women_i
N_i = n_participants_i
```

The model is conditional on the total number of participants:

```text
Y_i | N_i, p_i
```

Linear predictor:

```text
eta_i = alpha
      + beta_dist      * distance_log_std_i
      + beta_elev_gain * elevation_gain_log_std_i
      + beta_steep     * steepness_std_i
      + beta_alt       * altitude_std_i
      + beta_lon       * longitude_std_i
      + beta_lat       * latitude_std_i
      + beta_year      * year_std_i
```

Probability of female participation:

```text
p_i = inv_logit(eta_i)
```

Model 7 — Binomial logistic regression:

```text
Y_i ~ Binomial(N_i, p_i)
```

Model 8 — Beta-Binomial logistic regression:

```text
Y_i ~ BetaBinomial(N_i, p_i * phi, (1 - p_i) * phi)
phi = exp(log_phi)
```

The Beta-Binomial model adds overdispersion. Large `phi` values make the model close to a Binomial model. Smaller `phi` values indicate stronger race-to-race heterogeneity in female participation.

---

## Prior Choices

### Priors for mean finish time models

The following priors match the current Stan files `model1_normal.stan` and `model2_student_t.stan`:

| Parameter | Prior | Interpretation |
|---|---|---|
| `alpha` | `Normal(2.0, 1.0)` | broad intercept prior on log-hour scale |
| `beta_dist` | `Normal(0.7, 0.4)` | positive distance effect |
| `beta_elev` | `Normal(0.2, 0.3)` | positive but uncertain elevation-gain effect |
| `beta_steep` | `Normal(0.15, 0.2)` | weak-to-moderate positive steepness effect |
| `beta_alt` | `Normal(0.05, 0.1)` | weak positive altitude effect |
| `sigma` | `Normal(0, 0.3)`, constrained `sigma > 0` | residual scale on log-time scale |
| `nu_minus_two` | `Gamma(2, 0.1)` | Student-t tail parameter, with `nu > 2` |

### Priors for winning time models

The following priors match `model3_winning_log_normal.stan` and `model4_winning_log_student_t.stan`:

| Parameter | Prior | Interpretation |
|---|---|---|
| `alpha` | `Normal(0.0, 1.0)` | broad intercept prior on log winning-time scale |
| `beta_dist` | `Normal(0.7, 0.4)` | positive distance effect |
| `beta_elev` | `Normal(0.2, 0.3)` | positive but uncertain elevation-gain effect |
| `beta_steep` | `Normal(0.15, 0.2)` | weak-to-moderate positive steepness effect |
| `beta_alt` | `Normal(0.05, 0.1)` | weak positive altitude effect |
| `sigma` | `Normal(0, 0.3)`, constrained `sigma > 0` | residual scale on log winning-time scale |
| `nu_minus_two` | `Gamma(2, 0.1)` | Student-t tail parameter, with `nu > 2` |

The winning-time report includes prior predictive checks on both log and hour scales. Negative values on the log scale do **not** mean negative winning times; they correspond to positive winning times below one hour. If substantial prior predictive mass appears below one hour, the intercept prior should be reviewed.

### Priors for female participation models

The following priors match `model7_female_binomial.stan` and `model8_female_beta_binomial.stan`:

| Parameter | Prior | Interpretation |
|---|---|---|
| `alpha` | `Normal(logit(0.25), 1.0)` | baseline female share around 25%, with broad uncertainty |
| `beta_dist` | `Normal(-0.10, 0.35)` | longer races may reduce female share |
| `beta_elev_gain` | `Normal(-0.05, 0.35)` | elevation gain effect weakly negative but uncertain |
| `beta_steep` | `Normal(-0.05, 0.35)` | steepness effect weakly negative but uncertain |
| `beta_alt` | `Normal(0.00, 0.25)` | weak altitude prior |
| `beta_lon` | `Normal(0.00, 0.30)` | weak longitude/geography prior |
| `beta_lat` | `Normal(0.00, 0.30)` | weak latitude/geography prior |
| `beta_year` | `Normal(0.20, 0.25)` | positive prior for increasing female participation over time |
| `log_phi` | `Normal(log(50), 1.0)` | overdispersion prior for Beta-Binomial model |

---

## Model Comparison Strategy

Models are compared within each task only:

| Task | Compared models |
|---|---|
| Mean Finish Time | Model 1 Normal vs Model 2 Student-t |
| Winning Time | Model 3 Normal vs Model 4 Student-t |
| Female Participation | Model 7 Binomial vs Model 8 Beta-Binomial |

The comparison uses:

| Criterion | Meaning |
|---|---|
| PSIS-LOO | approximate leave-one-out cross-validation |
| WAIC | widely applicable information criterion |
| ELPD | expected log predictive density; higher is better |
| Pareto-k | reliability diagnostic for PSIS-LOO |
| Posterior predictive checks | visual and numerical checks of replicated data |

The preferred model should have better predictive performance, acceptable Pareto-k diagnostics and better posterior predictive behaviour.

---

## Main Results

The exact numerical values can vary slightly after rerunning MCMC. The summary below reflects the current project interpretation and the results recorded in the report materials.

### Mean Finish Time

The mean-time workflow supports modelling finish time on the log scale. The response is positive and right-skewed on the original hour scale, while the log transformation produces a more suitable regression target.

Main result:

> **Model 2: Student-t log-scale regression** is preferred over the Normal baseline because it is more robust to atypical races and heavy-tailed residuals.

The most important interpretation pattern is:

- distance is the dominant positive predictor of mean finish time;
- elevation gain and steepness increase expected finish time;
- altitude has a weaker effect after controlling for the other predictors;
- Student-t residuals better handle outlying or unusual races.

### Winning Time

The winning-time workflow mirrors the mean-time workflow but uses:

```text
log_winning_time = log(Winning Time)
```

Representative model-comparison results recorded in the previous README/report materials:

| Quantity | Model 3: Normal | Model 4: Student-t |
|---|---:|---:|
| ELPD LOO | 1394.78 | 1540.87 |
| LOO model weight | 0.042 | 0.958 |
| Max Pareto-k | 0.13 | 0.19 |
| Residual scale `sigma` | approx. 0.183 | approx. 0.146 |

Main result:

> **Model 4: Student-t log-scale regression** is preferred for winning time.

The Student-t likelihood improves predictive performance and handles unusual races more robustly. The longest `100M` races can still be difficult to reproduce perfectly, suggesting that category-level or hierarchical effects would be useful extensions.

### Female Participation

Female participation is modelled as a bounded count, not as an unconstrained continuous variable. This is important because:

```text
0 <= n_women <= n_participants
```

Observed female participation decreases with race category distance. Approximate category-level female shares recorded in the project materials are:

| Race category | Approx. female share |
|---|---:|
| `20K` | 31.0% |
| `50K` | 22.1% |
| `100K` | 17.1% |
| `100M` | 13.6% |

Representative model-comparison results recorded in the previous README/report materials:

| Quantity | Model 7: Binomial | Model 8: Beta-Binomial |
|---|---:|---:|
| ELPD LOO | -36,294 | -18,565 |
| LOO model weight | approx. 0.007 | approx. 0.993 |

Main result:

> **Model 8: Beta-Binomial logistic regression** is preferred for female participation.

The Beta-Binomial model captures overdispersion: races with similar observed covariates can still have substantially different female participation rates.

Main posterior interpretation:

- distance has a strong negative effect on female participation odds;
- steepness also tends to reduce female participation odds;
- year has a positive effect, indicating increasing female participation over time;
- altitude has no strong independent effect after controlling for distance, elevation, steepness, geography and year;
- the overdispersion parameter confirms meaningful race-to-race heterogeneity.

---

## Generated Outputs

The final report notebooks generate figures and cached model objects.

### Mean finish time report

Default figure directory:

```text
report_figures/
```

Typical outputs:

```text
missing_values.png
response_distributions.png
predictor_preprocessing.png
scatter_response_vs_predictors.png
correlation_matrix.png
model_diagram.png
prior_distributions.png
prior_predictive_*.png
trace_model1.png
trace_model2.png
posterior_model1.png
posterior_model2.png
comparison_loo.png
comparison_waic.png
pareto_k_diagnostics.png
joint_ppc_model1_model2.png
```

### Winning time report

Default figure directory:

```text
report_figures_winning_time/
```

Typical outputs:

```text
winning_response_distributions.png
winning_predictor_preprocessing.png
winning_scatter_response_vs_predictors.png
winning_correlation_matrix.png
winning_model_diagram.png
winning_prior_distributions.png
winning_prior_predictive_*.png
model3_winning_time_ppc.png
model4_winning_time_ppc.png
winning_time_joint_ppc_model3_model4.png
winning_time_pareto_k_diagnostics.png
winning_time_final_model_prediction.png
```

Cached model outputs:

```text
idata_winning_log_normal.nc
idata_winning_log_student_t.nc
```

### Female participation report

Default figure directory:

```text
report_figures_female_participation/
```

Typical outputs:

```text
female_response_distributions.png
female_predictor_distributions.png
female_share_vs_predictors.png
female_correlation_matrix.png
female_model_diagram.png
female_prior_distributions.png
female_prior_predictive_*.png
trace_binomial.png
trace_beta_binomial.png
posterior_odds_ratio_comparison.png
ppc_weighted_share.png
ppc_ecdf_binomial.png
ppc_ecdf_beta_binomial.png
ppc_by_category.png
comparison_loo.png
comparison_waic.png
comparison_pareto_k.png
final_model_distance_effect.png
final_model_year_effect.png
```

Cached model outputs:

```text
idata_female_binomial.nc
idata_female_beta_binomial.nc
```

---

## How to Run

Recommended order:

```text
1. final_bayesian_utmb_report_katex_fixed.ipynb
2. final_bayesian_utmb_winning_time_report_text_fixed.ipynb
3. final_bayesian_utmb_female_participation_report.ipynb
```

The notebooks are designed to be run from the repository root.

Place the raw dataset in the repository root or in a `data/` directory:

```text
utmb-race-data-sheet.csv
```

The notebooks can also use processed files if they exist:

```text
utmb_processed.csv
utmb_female_participation_processed.csv
```

The final notebooks use a subset for posterior fitting by default:

```text
MODEL_SAMPLE_N = 5000
```

This keeps generated quantities, posterior predictive arrays and pointwise log-likelihood matrices manageable.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/TomaszBryndza/utmb_race_finish_times_bayesian_analysis.git
cd utmb_race_finish_times_bayesian_analysis
```

### 2. Create a Python environment

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
```

or using Conda:

```bash
conda create -n utmb-bayes python=3.11 -y
conda activate utmb-bayes
```

### 3. Install Python dependencies

```bash
pip install numpy pandas scipy matplotlib arviz cmdstanpy kagglehub jupyter ipykernel ipywidgets tqdm
```

### 4. Install CmdStan

```bash
python -m cmdstanpy.install_cmdstan
```

CmdStan requires a working C++ toolchain.

### 5. Register a Jupyter kernel

```bash
python -m ipykernel install --user --name utmb-bayes --display-name "Python (UTMB Bayes)"
```

### 6. Start Jupyter

```bash
jupyter lab
```

or:

```bash
jupyter notebook
```

---

## Reproducibility Notes

- The notebooks set fixed random seeds, but MCMC results can still vary slightly across machines and package versions.
- Some reports use random subsampling for computational feasibility.
- Stan models may be recompiled automatically depending on the local CmdStan version.
- Expensive posterior objects should be cached as ArviZ NetCDF files.
- If preprocessing changes, rerun all downstream reports.
- Model comparison should always be based on pointwise `log_lik` generated in Stan.
- Posterior predictive checks should be interpreted alongside LOO/WAIC, not replaced by them.

---

## Troubleshooting Notes

### Negative values on log-scale plots

Negative values on axes such as:

```text
log(Winning Time [hours])
```

are not negative times. They correspond to positive times below one hour. For example:

```text
log(0.5) = -0.693
```

However, if a prior predictive check assigns substantial probability to winning times below one hour for ultra-trail races, the intercept prior should be reviewed.

### ArviZ posterior predictive dimensions

ArviZ expects posterior predictive arrays in the shape:

```text
chain x draw x observation
```

If an array has shape:

```text
draw x observation
```

add a fake chain dimension before creating an `InferenceData` object:

```python
time_rep = time_rep[None, :, :]
```

### Matplotlib `tight_layout` and colorbar issue

If `plt.tight_layout()` fails after creating a colorbar, use manual spacing instead:

```python
fig.subplots_adjust(left=0.25, bottom=0.30, right=0.88, top=0.90)
```

or create the figure with `constrained_layout=True` before adding axes.

### KaTeX / Markdown rendering

For markdown equations in notebooks, prefer:

```text
$$
...
$$
```

instead of `\[ ... \]` when the notebook renderer has KaTeX compatibility issues.

---

## Limitations and Future Work

Main limitations:

- the analysis uses race-level aggregated data, not individual runner-level data;
- weather, trail technicality, surface type and race-day conditions are not included;
- predictors such as distance, elevation gain and steepness are correlated;
- current models do not include hierarchical effects for race category, country or event family;
- some posterior inference uses subsampling for computational reasons;
- prior choices should be periodically rechecked with prior predictive simulations.

Recommended future extensions:

1. Add varying intercepts by `Race Category`.
2. Add varying intercepts by `Country` or event family.
3. Add nonlinear effects using splines or Gaussian processes.
4. Include weather, terrain technicality and aid-station structure if available.
5. Add category-level effects explicitly to female participation models.
6. Save all fitted models as ArviZ `InferenceData` objects.
7. Refactor the repository into `data/`, `models/`, `notebooks/`, `figures/` and `outputs/` folders.
8. Compare models using out-of-sample validation on held-out races or years.

---

## Final Project Conclusion

Bayesian modelling is well suited for UTMB race-level analysis because it combines interpretable assumptions, uncertainty quantification and predictive checking.

Across the three modelling tasks:

| Task | Preferred model | Reason |
|---|---|---|
| Mean Finish Time | Model 2: Student-t log regression | robust to heavy-tailed residuals and atypical races |
| Winning Time | Model 4: Student-t log regression | better predictive performance than Normal baseline |
| Female Participation | Model 8: Beta-Binomial logistic regression | captures overdispersion in bounded participation counts |

The key statistical lesson is that the likelihood matters. Matching the likelihood to the response type — positive continuous time, heavy-tailed log-time residuals or bounded counts — leads to more realistic uncertainty, better posterior predictive behaviour and stronger out-of-sample predictive performance.
