# Bayesian Analysis of UTMB Ultra-Trail Race Data

This repository contains a complete Bayesian workflow for analysing UTMB World Series race-level data. The project models three related phenomena:

1. **Mean Finish Time** — the average race finish time in hours.
2. **Winning Time** — the finish time achieved by the race winner in hours.
3. **Female Participation** — the number and share of female participants in a race.

The analysis is implemented in **Python**, **Jupyter Notebook**, **Stan**, **CmdStanPy** and **ArviZ**. The project follows the full Bayesian workflow: problem formulation, exploratory data analysis, feature engineering, prior design, prior predictive checks, posterior sampling, sampling diagnostics, posterior predictive checks and model comparison using **PSIS-LOO** and **WAIC**.

The main modelling principle is:

> Use a likelihood that matches the mathematical nature of the response variable.

Positive race times are modelled on the logarithmic scale. Female participation is modelled as a bounded count conditional on the total number of participants.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Current Final Reports](#current-final-reports)
- [Research Questions](#research-questions)
- [Repository Structure](#repository-structure)
- [Feature Engineering](#feature-engineering)
- [Model Specifications](#model-specifications)
- [Prior Choices](#prior-choices)
- [Posterior Inference and Diagnostics](#posterior-inference-and-diagnostics)
- [Model Comparison Strategy](#model-comparison-strategy)
- [Main Results](#main-results)
- [Generated Figures and Outputs](#generated-figures-and-outputs)
- [How to Run](#how-to-run)
- [Installation](#installation)
- [Reproducibility Notes](#reproducibility-notes)
- [Troubleshooting Notes](#troubleshooting-notes)
- [Limitations and Future Work](#limitations-and-future-work)
- [Final Project Conclusion](#final-project-conclusion)

---

## Project Overview

Ultra-trail race performance depends on multiple interacting factors: distance, elevation gain, route steepness, altitude, race category, geography, race year and unobserved race-specific conditions. Races with similar distance and elevation can still differ substantially because of weather, technical terrain, aid-station structure, participant profile and local event characteristics.

A Bayesian approach is useful because it provides:

- interpretable prior assumptions;
- full posterior uncertainty for parameters;
- posterior predictive distributions instead of point predictions only;
- robust likelihoods for atypical races and heavy-tailed residuals;
- principled model comparison using expected out-of-sample predictive performance.

The project compares baseline models with more flexible alternatives:

- **Normal vs Student-t** likelihoods for log-scale race time models;
- **Binomial vs Beta-Binomial** likelihoods for female participation counts.

---

## Dataset

The project uses race-level UTMB World Series data from Kaggle:

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

The analysis is **race-level**, not runner-level. Each row represents an aggregated race observation.

---

## Current Final Reports

The current final report notebooks on the `main` branch are:

| Report notebook | Topic | Compared models | Figure directory |
|---|---|---|---|
| `final_bayesian_utmb_report_tb.ipynb` | Mean finish time | Model 1 Normal vs Model 2 Student-t | `report_figures/` |
| `final_bayesian_utmb_winning_time_report.ipynb` | Winning time | Model 3 Normal vs Model 4 Student-t | `report_figures_winning_time/` |
| `final_bayesian_utmb_female_participation_report.ipynb` | Female participation | Model 7 Binomial vs Model 8 Beta-Binomial | `report_figures_female_participation/` |

These notebooks are report-oriented. They are intended to be cleaner than the original exploratory notebooks and include EDA, model formulation, priors, prior predictive checks, diagnostics, posterior analysis, posterior predictive checks and model comparison.

Development notebooks are still useful as project history:

| Notebook | Role |
|---|---|
| `01_problem_formulation.ipynb` | Initial data loading, cleaning and problem formulation |
| `02_model_specification_priors.ipynb` | Initial prior specification and prior predictive experiments |
| `03_posterior_model1_normal.ipynb` | Posterior analysis for the mean-time Normal model |
| `04_posterior_model2_student_t.ipynb` | Posterior analysis for the mean-time Student-t model |
| `05_model_comparison.ipynb` | Earlier mean-time model comparison |
| `06_winning_time_log_modeling.ipynb` | Development notebook for log-scale winning-time models |
| `07_female_participation_modeling.ipynb` | Development notebook for female participation models |
| `08_final_bayesian_project_report.ipynb` | Earlier combined final-report draft |

---

## Research Questions

The project addresses the following questions:

1. How do distance, elevation gain, steepness and altitude influence expected mean finish time?
2. Is log-scale modelling appropriate for positive and right-skewed race times?
3. Does a Student-t likelihood improve predictive performance over a Normal likelihood for race time models?
4. How do distance, elevation, steepness, altitude, geography and year affect female participation?
5. Is ordinary Binomial variation sufficient for female participation counts, or is overdispersion present?
6. Which model should be preferred for each modelling task according to PSIS-LOO, WAIC, Pareto-k diagnostics and posterior predictive checks?

---

## Repository Structure

Recommended interpretation of the current repository structure:

```text
.
├── README.md
├── utmb-race-data-sheet.csv
├── utmb_processed.csv
├── utmb_female_participation_processed.csv
│
├── final_bayesian_utmb_report_tb.ipynb
├── final_bayesian_utmb_winning_time_report.ipynb
├── final_bayesian_utmb_female_participation_report.ipynb
│
├── 01_problem_formulation.ipynb
├── 02_model_specification_priors.ipynb
├── 03_posterior_model1_normal.ipynb
├── 04_posterior_model2_student_t.ipynb
├── 05_model_comparison.ipynb
├── 06_winning_time_log_modeling.ipynb
├── 07_female_participation_modeling.ipynb
├── 08_final_bayesian_project_report.ipynb
│
├── model1_normal.stan
├── model2_student_t.stan
├── model3_winning_log_normal.stan
├── model4_winning_log_student_t.stan
├── model7_female_binomial.stan
├── model8_female_beta_binomial.stan
│
├── report_figures/
├── report_figures_winning_time/
└── report_figures_female_participation/
```

Some compiled Stan executables or generated files may also be present. They are not conceptually required to understand the project and can usually be regenerated locally.

---

## Feature Engineering

Continuous predictors are transformed and standardized before modelling. Standardization makes priors easier to specify and makes coefficients comparable across predictors.

### Time models

Used for **Mean Finish Time** and **Winning Time**:

| Predictor | Construction | Purpose |
|---|---|---|
| `distance_log_std` | standardized `log(Distance)` | captures multiplicative distance effects |
| `elevation_log_std` | standardized `log1p(Elevation Gain)` | handles skewed positive elevation gain |
| `steepness_std` | standardized `log1p(Elevation Gain / Distance)` | separates route steepness from total elevation |
| `altitude_std` | standardized `log1p(Elevation)` or standardized altitude depending on notebook preprocessing | captures altitude-related difficulty |

For time models, the responses are log-transformed:

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

The female participation model uses course-difficulty predictors and adds geography and year:

| Predictor | Construction | Purpose |
|---|---|---|
| `distance_log_std` | standardized `log(Distance)` | course length effect |
| `elevation_gain_log_std` | standardized `log1p(Elevation Gain)` | elevation gain effect |
| `steepness_std` | standardized `log1p(Elevation Gain / Distance)` | steepness effect |
| `altitude_std` | standardized altitude/elevation variable | altitude effect |
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
phi = exp(log_phi)
Y_i ~ BetaBinomial(N_i, p_i * phi, (1 - p_i) * phi)
```

The Beta-Binomial model adds overdispersion. Large `phi` values make it close to the Binomial model. Smaller `phi` values indicate stronger race-to-race heterogeneity in female participation.

---

## Prior Choices

This section lists the priors exactly as implemented in the current Stan model blocks. The model block is treated as the source of truth if an older notebook comment or markdown note is inconsistent with it.

Stan uses the `Gamma(shape, rate)` parameterization. Therefore, `Gamma(2, 0.1)` has mean `20`, and because the Student-t models define `nu = 2 + nu_minus_two`, the prior mean of `nu` is approximately `22`.

### Mean finish time models

Target:

```text
log_time = log(Mean Finish Time)
```

#### Model 1 — Normal log-time regression

Implemented in `model1_normal.stan`.

| Parameter | Prior implemented in Stan | Notes |
|---|---|---|
| `alpha` | `Normal(2.0, 1.0)` | intercept on log-hour scale |
| `beta_dist` | `Normal(0.7, 0.4)` | effect of standardized log distance |
| `beta_elev` | `Normal(0.2, 0.3)` | effect of standardized log elevation gain |
| `beta_steep` | `Normal(0.15, 0.2)` | effect of standardized route steepness |
| `beta_alt` | `Normal(0.05, 0.1)` | effect of standardized altitude |
| `sigma` | `Normal(0, 0.3)`, with `sigma > 0` | half-normal residual scale on log-time scale |

#### Model 2 — Student-t log-time regression

Implemented in `model2_student_t.stan`. Regression priors are identical to Model 1; the additional prior is for the Student-t tail parameter.

| Parameter | Prior implemented in Stan | Notes |
|---|---|---|
| `alpha` | `Normal(2.0, 1.0)` | intercept on log-hour scale |
| `beta_dist` | `Normal(0.7, 0.4)` | effect of standardized log distance |
| `beta_elev` | `Normal(0.2, 0.3)` | effect of standardized log elevation gain |
| `beta_steep` | `Normal(0.15, 0.2)` | effect of standardized route steepness |
| `beta_alt` | `Normal(0.05, 0.1)` | effect of standardized altitude |
| `sigma` | `Normal(0, 0.3)`, with `sigma > 0` | half-normal residual scale on log-time scale |
| `nu_minus_two` | `Gamma(2, 0.1)` | `nu = 2 + nu_minus_two`, so `nu > 2` |

### Winning time models

Target:

```text
log_winning_time = log(Winning Time)
```

#### Model 3 — Normal log-winning-time regression

Implemented in `model3_winning_log_normal.stan`.

| Parameter | Prior implemented in Stan | Notes |
|---|---|---|
| `alpha` | `Normal(1.7, 0.2)` | intercept on log-hour scale; `exp(1.7) ≈ 5.47` hours for an average standardized race |
| `beta_dist` | `Normal(0.7, 0.4)` | effect of standardized log distance |
| `beta_elev` | `Normal(0.2, 0.3)` | effect of standardized log elevation gain |
| `beta_steep` | `Normal(0.15, 0.2)` | effect of standardized route steepness |
| `beta_alt` | `Normal(0.05, 0.1)` | effect of standardized altitude |
| `sigma` | `Normal(0, 0.3)`, with `sigma > 0` | half-normal residual scale on log-winning-time scale |

#### Model 4 — Student-t log-winning-time regression

Implemented in `model4_winning_log_student_t.stan`. Regression priors are identical to Model 3; the additional prior is for the Student-t tail parameter.

| Parameter | Prior implemented in Stan | Notes |
|---|---|---|
| `alpha` | `Normal(1.7, 0.2)` | intercept on log-hour scale; `exp(1.7) ≈ 5.47` hours for an average standardized race |
| `beta_dist` | `Normal(0.7, 0.4)` | effect of standardized log distance |
| `beta_elev` | `Normal(0.2, 0.3)` | effect of standardized log elevation gain |
| `beta_steep` | `Normal(0.15, 0.2)` | effect of standardized route steepness |
| `beta_alt` | `Normal(0.05, 0.1)` | effect of standardized altitude |
| `sigma` | `Normal(0, 0.3)`, with `sigma > 0` | half-normal residual scale on log-winning-time scale |
| `nu_minus_two` | `Gamma(2, 0.1)` | `nu = 2 + nu_minus_two`, so `nu > 2` |

The winning-time intercept prior is intentionally more concentrated than the earlier broad experimental prior. It avoids placing excessive prior predictive mass on implausibly short winning times while still allowing the predictors and residual scale to express uncertainty.

### Female participation models

Target:

```text
n_women | n_participants, p
```

The linear predictor is on the log-odds scale, and `p = inv_logit(eta)`.

#### Model 7 — Binomial logistic regression

Implemented in `model7_female_binomial.stan`.

| Parameter | Prior implemented in Stan | Notes |
|---|---|---|
| `alpha` | `Normal(-1.0986122886681098, 1.0)` | equivalent to `Normal(logit(0.25), 1.0)`; baseline female share around 25% |
| `beta_dist` | `Normal(-0.10, 0.35)` | effect of standardized log distance on log-odds |
| `beta_elev_gain` | `Normal(-0.05, 0.35)` | effect of standardized log elevation gain on log-odds |
| `beta_steep` | `Normal(-0.05, 0.35)` | effect of standardized steepness on log-odds |
| `beta_alt` | `Normal(0.00, 0.25)` | effect of standardized altitude on log-odds |
| `beta_lon` | `Normal(0.00, 0.30)` | longitude/geographic effect on log-odds |
| `beta_lat` | `Normal(0.00, 0.30)` | latitude/geographic effect on log-odds |
| `beta_year` | `Normal(0.20, 0.25)` | temporal trend in female participation log-odds |

#### Model 8 — Beta-Binomial logistic regression

Implemented in `model8_female_beta_binomial.stan`. Regression priors are identical to Model 7; the additional prior is for the overdispersion parameter.

| Parameter | Prior implemented in Stan | Notes |
|---|---|---|
| `alpha` | `Normal(-1.0986122886681098, 1.0)` | equivalent to `Normal(logit(0.25), 1.0)`; baseline female share around 25% |
| `beta_dist` | `Normal(-0.10, 0.35)` | effect of standardized log distance on log-odds |
| `beta_elev_gain` | `Normal(-0.05, 0.35)` | effect of standardized log elevation gain on log-odds |
| `beta_steep` | `Normal(-0.05, 0.35)` | effect of standardized steepness on log-odds |
| `beta_alt` | `Normal(0.00, 0.25)` | effect of standardized altitude on log-odds |
| `beta_lon` | `Normal(0.00, 0.30)` | longitude/geographic effect on log-odds |
| `beta_lat` | `Normal(0.00, 0.30)` | latitude/geographic effect on log-odds |
| `beta_year` | `Normal(0.20, 0.25)` | temporal trend in female participation log-odds |
| `log_phi` | `Normal(log(50), 1.0)` | overdispersion prior; `phi = exp(log_phi)` |

---

## Posterior Inference and Diagnostics

Each final report includes:

- trace plots for important parameters;
- rank plots or chain-mixing diagnostics where useful;
- R-hat and effective sample size summaries;
- divergence checks;
- energy/BFMI diagnostics where relevant;
- posterior parameter distributions;
- forest plots with HDI intervals;
- posterior predictive checks;
- residual or calibration plots where meaningful.

For time models, posterior predictions are inspected on both:

```text
log-time scale
hour scale after exp(...)
```

For female participation, posterior predictions are inspected as:

```text
n_women replicated counts
race-level female share = n_women / n_participants
weighted female share = sum(n_women) / sum(n_participants)
```

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

A preferred model should have better predictive performance, acceptable Pareto-k diagnostics and better posterior predictive behaviour.

---

## Main Results

Numerical values can vary slightly after rerunning MCMC because the notebooks use probabilistic sampling. The summary below reflects the current report interpretation.

### Mean Finish Time

The mean-time workflow supports modelling finish time on the log scale. The original hour-scale response is positive and right-skewed, while the log transformation creates a more suitable regression target.

Main interpretation pattern:

- distance is the dominant positive predictor of mean finish time;
- elevation gain and route steepness increase expected finish time;
- altitude has a weaker independent effect after controlling for the other predictors;
- the Student-t likelihood is more robust to atypical races and heavy-tailed residuals.

Main result:

> **Model 2: Student-t log-scale regression** is preferred over the Normal baseline for mean finish time.

### Winning Time

The winning-time workflow mirrors the mean-time workflow but uses:

```text
log_winning_time = log(Winning Time)
```

The raw `Winning Time` distribution is right-skewed, while `log_winning_time` is better suited for regression.

Representative model-comparison results recorded in the project materials:

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

Female participation is modelled as a bounded count, not as an unconstrained continuous percentage:

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

Representative model-comparison results recorded in the project materials:

| Quantity | Model 7: Binomial | Model 8: Beta-Binomial |
|---|---:|---:|
| ELPD LOO | -36,294 | -18,565 |
| LOO model weight | approx. 0.007 | approx. 0.993 |

Main posterior interpretation:

- distance has a strong negative effect on female participation odds;
- steepness also tends to reduce female participation odds;
- year has a positive effect, indicating increasing female participation over time;
- altitude has no strong independent effect after controlling for distance, elevation, steepness, geography and year;
- the overdispersion parameter confirms meaningful race-to-race heterogeneity.

Main result:

> **Model 8: Beta-Binomial logistic regression** is preferred for female participation.

---

## Generated Figures and Outputs

The final notebooks save report figures into dedicated directories.

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
prior_predictive_log_scale_normal.png
prior_predictive_log_scale_student-t.png
prior_predictive_hour_scale_ecdf_normal.png
prior_predictive_hour_scale_ecdf_student-t.png
model1_trace.png
model2_trace.png
model1_ppc.png
model2_ppc.png
joint_ppc_model1_model2.png
model_comparison_loo_waic.png
pareto_k_diagnostics.png
final_winner_prediction.png
```

### Winning time report

Default figure directory:

```text
report_figures_winning_time/
```

Typical outputs:

```text
winning_target_distribution.png
winning_scatter_response_vs_predictors.png
winning_correlation_matrix.png
winning_model_diagram.png
winning_prior_distributions.png
winning_prior_predictive_log_normal.png
winning_prior_predictive_log_student-t.png
winning_prior_predictive_ecdf_normal.png
winning_prior_predictive_ecdf_student-t.png
model3_winning_time_ppc.png
model4_winning_time_ppc.png
winning_time_joint_ppc_model3_model4.png
winning_time_pareto_k_diagnostics.png
winning_loo_compare.png
winning_waic_compare.png
winning_prediction_vs_distance_final_student-t_model.png
```

### Female participation report

Default figure directory:

```text
report_figures_female_participation/
```

Typical outputs:

```text
eda_response_distributions.png
eda_predictor_distributions.png
eda_share_vs_predictors.png
eda_correlation_matrix.png
model_dependency_diagram.png
prior_parameter_distributions.png
prior_predictive_female_share_ecdf.png
prior_predictive_weighted_share.png
diagnostics_binomial_trace.png
diagnostics_beta_binomial_trace.png
posterior_odds_ratio_comparison.png
ppc_weighted_share.png
ppc_ecdf_binomial.png
ppc_ecdf_beta_binomial.png
ppc_by_category.png
female_participation_joint_ppc_model7_model8.png
female_participation_pareto_k_diagnostics.png
comparison_loo.png
comparison_waic.png
final_model_distance_effect.png
final_model_year_effect.png
```

Optional cached model outputs may include:

```text
idata_model1_normal.nc
idata_model2_student_t.nc
idata_winning_log_normal.nc
idata_winning_log_student_t.nc
idata_female_binomial.nc
idata_female_beta_binomial.nc
```

If these files are absent, the notebooks fit the Stan models again.

---

## How to Run

Recommended order for the final reports:

```text
1. final_bayesian_utmb_report_tb.ipynb
2. final_bayesian_utmb_winning_time_report.ipynb
3. final_bayesian_utmb_female_participation_report.ipynb
```

Run notebooks from the repository root so that relative paths to data, Stan files and figure directories work correctly.

Place the raw dataset in the repository root or in a `data/` directory:

```text
utmb-race-data-sheet.csv
```

The reports can also use processed files if they exist:

```text
utmb_processed.csv
utmb_female_participation_processed.csv
```

The final notebooks use a reproducible sample for posterior fitting by default:

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
- Posterior predictive checks should be interpreted together with LOO/WAIC, not replaced by them.

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

### ArviZ PPC variable names

When using `az.plot_ppc`, the observed variable and posterior predictive variable names must match through `data_pairs`. For example:

```python
az.plot_ppc(
    idata,
    var_names=["winning_time"],
    data_pairs={"winning_time": "winning_time_rep"},
)
```

If the object also contains log-scale variables, explicitly set `var_names` to avoid ArviZ selecting the wrong observed variable.

### Matplotlib `tight_layout` and colorbar issue

If `plt.tight_layout()` fails after creating a colorbar, use manual spacing instead:

```python
fig.subplots_adjust(left=0.25, bottom=0.30, right=0.88, top=0.90)
```

or create the figure with `constrained_layout=True` before adding axes.

### KaTeX / Markdown rendering

For markdown equations in notebooks, prefer:

```text
$$ ... $$
```

instead of:

```text
\[ ... \]
```

when the notebook renderer has KaTeX compatibility issues.

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
8. Compare models using held-out races or held-out years.

---

## Final Project Conclusion

Bayesian modelling is well suited for UTMB race-level analysis because it combines interpretable assumptions, uncertainty quantification and predictive checking.

Across the three modelling tasks:

| Task | Preferred model | Reason |
|---|---|---|
| Mean Finish Time | Model 2: Student-t log regression | robust to heavy-tailed residuals and atypical races |
| Winning Time | Model 4: Student-t log regression | better predictive performance than the Normal baseline |
| Female Participation | Model 8: Beta-Binomial logistic regression | captures overdispersion in bounded participation counts |

The key statistical lesson is that the likelihood matters. Matching the likelihood to the response type — positive continuous time, heavy-tailed log-time residuals or bounded counts — leads to more realistic uncertainty, better posterior predictive behaviour and stronger out-of-sample performance.
