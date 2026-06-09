# Bayesian Analysis of UTMB Ultra-Trail Race Data

This repository contains a complete Bayesian workflow for analysing UTMB World Series race-level data. The project models three related phenomena:

1. **Mean Finish Time** — average race finish time.
2. **Winning Time** — time achieved by the race winner.
3. **Female Participation** — number and share of female participants.

The analysis is implemented in Python, Jupyter Notebook and Stan/CmdStanPy. The project follows a full Bayesian workflow: problem formulation, exploratory data analysis, prior design, prior predictive checks, posterior sampling, posterior diagnostics, posterior predictive checks and model comparison using PSIS-LOO and WAIC.

The main modelling principle is:

> Use a likelihood that matches the mathematical nature of the response variable.

Therefore, positive continuous race times are modelled on the logarithmic scale, while female participation is modelled as a bounded count conditioned on the total number of participants.

---

## Table of Contents

- [Project Motivation](#project-motivation)
- [Dataset](#dataset)
- [Research Questions](#research-questions)
- [Repository Structure](#repository-structure)
- [Modelling Tasks](#modelling-tasks)
- [Feature Engineering](#feature-engineering)
- [Model Specifications](#model-specifications)
- [Prior Choices](#prior-choices)
- [Model Comparison Strategy](#model-comparison-strategy)
- [Main Results](#main-results)
- [How to Run](#how-to-run)
- [Installation](#installation)
- [Generated Figures](#generated-figures)
- [Reproducibility Notes](#reproducibility-notes)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Project Motivation

Ultra-trail race performance is shaped by several interacting factors: distance, elevation gain, route steepness, altitude, geographical location, race category and year. A deterministic model is insufficient because races with similar distance and elevation can still differ significantly due to route technicality, weather, race organization, participant profile and other unobserved factors.

A Bayesian approach is useful because it provides:

- full posterior uncertainty for model parameters;
- interpretable prior assumptions;
- posterior predictive distributions instead of point predictions only;
- explicit treatment of outliers through robust likelihoods;
- principled model comparison through approximate out-of-sample predictive performance.

The project compares simple baseline models with more robust or flexible alternatives. This makes it possible to evaluate whether additional likelihood flexibility is justified by the data.

---

## Dataset

The project uses UTMB World Series race-level data from Kaggle:

```text
mgpoirot/utmb-world-race-daa
```

The preprocessing focuses on the four main UTMB race categories:

```text
20K, 50K, 100K, 100M
```

The cleaned dataset is saved as:

```text
utmb_processed.csv
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
| `N Participants` | Total number of participants | female participation model |
| `N Women` | Number of female participants | female participation model |
| `Year` | Race year | female participation model |
| `Longitude`, `Latitude` | Race/event coordinates | female participation model |

The analysis is race-level, not runner-level. Each row represents an aggregated race observation.

---

## Research Questions

The project addresses the following questions:

1. How do distance, elevation gain, steepness and altitude influence expected race finish times?
2. Is log-scale modelling appropriate for positive and right-skewed race times?
3. Does a Student-t likelihood improve predictive performance over a Normal likelihood for time models?
4. How do distance, elevation, geography and year affect female participation?
5. Is ordinary Binomial variation sufficient for female participation counts, or is overdispersion present?
6. Which model should be preferred for each modelling task according to LOO and WAIC?

---

## Repository Structure

Recommended final project structure:

```text
.
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
├── prior_predictive.stan
│
├── utmb_processed.csv
├── README.md
├── PROJECT_DESCRIPTION.md
│
├── fig*.png
├── fig_wt_*.png
├── fig_female_*.png
└── plot*.png
```

### Notebook roles

| Notebook | Purpose |
|---|---|
| `01_problem_formulation.ipynb` | Data loading, cleaning, preprocessing and exploratory analysis |
| `02_model_specification_priors.ipynb` | Model definitions, prior rationale and prior predictive checks |
| `03_posterior_model1_normal.ipynb` | Posterior analysis of the Normal mean-time model |
| `04_posterior_model2_student_t.ipynb` | Posterior analysis of the Student-t mean-time model |
| `05_model_comparison.ipynb` | LOO/WAIC comparison for mean finish time models |
| `06_winning_time_log_modeling.ipynb` | Log-scale Bayesian workflow for winning time |
| `07_female_participation_modeling.ipynb` | Bayesian modelling of female participation counts |
| `08_final_bayesian_project_report.ipynb` | Final condensed report combining the whole project |

---

## Modelling Tasks

## 1. Mean Finish Time

Target:

```text
log_time = log(Mean Finish Time)
```

Two models are compared:

| Model | Likelihood | Purpose |
|---|---|---|
| Model 1 | Normal | baseline log-time regression |
| Model 2 | Student-t | robust log-time regression |

The logarithmic transformation is used because finish time is positive and right-skewed. Back-transforming with `exp(...)` guarantees positive predicted times in hours.

---

## 2. Winning Time

Target:

```text
log_winning_time = log(Winning Time)
```

Two models are compared:

| Model | Likelihood | Purpose |
|---|---|---|
| Model 3 | Normal | baseline log-winning-time regression |
| Model 4 | Student-t | robust log-winning-time regression |

The final winning-time workflow mirrors the mean-time workflow. Earlier original-scale winning-time models may exist in the repository as legacy files, but the final recommended approach is the log-scale version.

---

## 3. Female Participation

Target:

```text
N Women
```

The target is a bounded count:

```text
0 <= N Women <= N Participants
```

Therefore, the model is formulated conditionally on the total number of race participants:

```text
N Women_i | N Participants_i, p_i
```

Two models are compared:

| Model | Likelihood | Purpose |
|---|---|---|
| Model 7A | Binomial | baseline bounded count model |
| Model 7B | Beta-Binomial | overdispersed bounded count model |

The Beta-Binomial model is included because race-level participation can vary more than expected under a simple Binomial model.

---

## Feature Engineering

All continuous predictors are transformed and standardized before modelling. This makes priors easier to specify and makes regression coefficients comparable.

### Time models

Predictors used for mean finish time and winning time:

| Predictor | Construction | Reason |
|---|---|---|
| `distance_log_std` | standardized `log(Distance)` | distance has a multiplicative effect on time |
| `elevation_log_std` | standardized `log(Elevation Gain + 1)` | elevation gain is positive and skewed |
| `steepness_std` | standardized `log1p(Elevation Gain / Distance)` | separates route steepness from total elevation |
| `altitude_std` | standardized altitude or `log1p(Elevation)` | altitude can affect performance |

### Female participation model

Additional predictors:

| Predictor | Reason |
|---|---|
| `longitude_std` | captures broad geographical differences |
| `latitude_std` | captures broad geographical differences |
| `year_std` | captures time trend in female participation |

---

## Model Specifications

## Mean Finish Time: Model 1 and Model 2

For race `i`:

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

Model 1:

```text
y_i ~ Normal(mu_i, sigma)
```

Model 2:

```text
y_i ~ Student_t(nu, mu_i, sigma)
nu = 2 + nu_minus_two
```

Predicted time in hours:

```text
time_mu = exp(mu)
time_rep = exp(log_time_rep)
```

---

## Winning Time: Model 3 and Model 4

For race `i`:

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

Model 3:

```text
y_i ~ Normal(mu_i, sigma)
```

Model 4:

```text
y_i ~ Student_t(nu, mu_i, sigma)
nu = 2 + nu_minus_two
```

Because the model is fitted on the log scale, each coefficient has a multiplicative interpretation after exponentiation:

```text
exp(beta)
```

This is the multiplicative change in predicted time for a one-standard-deviation increase in a predictor, holding the others fixed.

---

## Female Participation: Model 7A and Model 7B

Let:

```text
Y_i = N Women_i
N_i = N Participants_i
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

Model 7A — Binomial:

```text
Y_i ~ Binomial(N_i, p_i)
```

Model 7B — Beta-Binomial:

```text
Y_i ~ BetaBinomial(N_i, p_i * phi, (1 - p_i) * phi)
```

The parameter `phi` controls overdispersion. Large `phi` values make the Beta-Binomial behave similarly to the Binomial. Smaller `phi` values indicate stronger race-to-race heterogeneity.

---

## Prior Choices

## Priors for time models

The time models use weakly informative priors on the log-time scale.

| Parameter | Prior | Interpretation |
|---|---|---|
| `alpha` | `Normal(0, 1)` | weak intercept prior on log-hour scale |
| `beta_dist` | `Normal(0.7, 0.4)` | strong positive effect of race distance |
| `beta_elev` | `Normal(0.2, 0.3)` | positive but uncertain effect of elevation gain |
| `beta_steep` | `Normal(0.15, 0.2)` | weak-to-moderate positive effect of steepness |
| `beta_alt` | `Normal(0.05, 0.1)` | weak positive altitude effect |
| `sigma` | `Normal(0, 0.3)` with `sigma > 0` | residual scale on log-time scale |
| `nu_minus_two` | `Gamma(2, 0.1)` | Student-t tail parameter, with `nu > 2` |

The priors encode plausible domain assumptions while remaining broad enough for the data to dominate.

---

## Priors for female participation models

All predictors are standardized. Coefficients are interpreted as changes in log-odds per one-standard-deviation increase.

| Parameter | Prior | Interpretation |
|---|---|---|
| `alpha` | `Normal(logit(0.25), 1)` | baseline female share around 25%, with broad uncertainty |
| `beta_dist` | `Normal(-0.10, 0.35)` | longer races may reduce female share |
| `beta_elev_gain` | `Normal(-0.05, 0.35)` | elevation difficulty may reduce female share |
| `beta_steep` | `Normal(-0.05, 0.35)` | steepness effect uncertain, weakly negative |
| `beta_alt` | `Normal(0, 0.25)` | weak environmental prior |
| `beta_lon` | `Normal(0, 0.30)` | weak spatial prior |
| `beta_lat` | `Normal(0, 0.30)` | weak spatial prior |
| `beta_year` | `Normal(0.20, 0.25)` | female participation may increase over time |
| `log_phi` | `Normal(log(50), 1)` | weak prior for overdispersion |

Prior predictive checks are used to verify that the priors generate plausible participation rates before conditioning on the observed data.

---

## Model Comparison Strategy

Models are compared using:

| Criterion | Meaning |
|---|---|
| PSIS-LOO | approximate leave-one-out cross-validation |
| WAIC | widely applicable information criterion |
| ELPD | expected log predictive density; higher is better |
| Pareto-k | reliability diagnostic for PSIS-LOO |
| Posterior predictive checks | whether replicated data resemble observed data |

The comparison is made within each task only:

- Mean Finish Time: Normal vs Student-t.
- Winning Time: Normal vs Student-t.
- Female Participation: Binomial vs Beta-Binomial.

---

## Main Results

## Mean Finish Time

The mean-time workflow shows that the log-scale formulation is appropriate for positive, right-skewed race times. The Student-t likelihood is preferred over the Normal baseline because it handles atypical races and heavy-tailed residuals more robustly.

Main conclusion:

> For average race finish time, the Student-t log-scale regression is preferred over the Normal log-scale regression.

---

## Winning Time

The winning-time workflow also supports log-scale modelling. The raw `Winning Time` distribution is strongly right-skewed, while `log_winning_time` is substantially closer to symmetric.

Key posterior and model-comparison results:

| Quantity | Normal | Student-t |
|---|---:|---:|
| ELPD LOO | 1394.78 | 1540.87 |
| LOO model weight | 0.042 | 0.958 |
| Max Pareto-k | 0.13 | 0.19 |
| Residual scale `sigma` | approx. 0.183 | approx. 0.146 |

The Student-t model improves predictive performance and estimates a smaller residual scale. Its degrees-of-freedom parameter indicates moderate heavy-tailed behaviour.

Main conclusion:

> For winning time, the Student-t log-scale model is preferred. However, the longest `100M` races remain difficult to reproduce perfectly, suggesting that category-level or hierarchical effects would be a useful extension.

---

## Female Participation

Observed female participation decreases with race category distance:

| Race category | Approx. female share |
|---|---:|
| `20K` | 31.0% |
| `50K` | 22.1% |
| `100K` | 17.1% |
| `100M` | 13.6% |

Model comparison strongly favours the Beta-Binomial model:

| Quantity | Binomial | Beta-Binomial |
|---|---:|---:|
| ELPD LOO | -36,294 | -18,565 |
| LOO model weight | approx. 0.007 | approx. 0.993 |

The Beta-Binomial model captures substantial overdispersion. Races with similar observed covariates can still have very different female participation rates.

Main posterior conclusions from the preferred Beta-Binomial model:

- distance has a strong negative effect on female participation odds;
- steepness also has a negative effect;
- year has a positive effect, indicating increasing female participation over time;
- altitude has no strong independent effect after controlling for distance, elevation, steepness, geography and year;
- the overdispersion parameter confirms meaningful race-to-race heterogeneity.

Main conclusion:

> For female participation, the Beta-Binomial logistic regression is preferred over the Binomial model.

---

## How to Run

Run the notebooks in the following order:

```text
01_problem_formulation.ipynb
02_model_specification_priors.ipynb
03_posterior_model1_normal.ipynb
04_posterior_model2_student_t.ipynb
05_model_comparison.ipynb
06_winning_time_log_modeling.ipynb
07_female_participation_modeling.ipynb
08_final_bayesian_project_report.ipynb
```

The first notebook creates or refreshes `utmb_processed.csv`. Downstream notebooks assume that the processed dataset is available in the repository root.

The final notebook `08_final_bayesian_project_report.ipynb` is a compact report notebook. It documents the complete modelling workflow and summarizes the final results without repeating all expensive MCMC sampling.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/TomaszBryndza/utmb_race_finish_times_bayesian_analysis.git
cd utmb_race_finish_times_bayesian_analysis
git checkout unified_log_model
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

### 3. Install dependencies

```bash
pip install numpy pandas scipy matplotlib arviz cmdstanpy kagglehub jupyter ipykernel ipywidgets tqdm
```

### 4. Install CmdStan

```bash
python -m cmdstanpy.install_cmdstan
```

CmdStan requires a working C++ toolchain. On Linux/macOS, make sure `make` and a C++ compiler are installed. On Windows, a compatible CmdStan toolchain is required.

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

## Generated Figures

The project uses figures to answer specific modelling questions, not just to decorate the notebook.

Important plot types:

| Plot type | Purpose |
|---|---|
| Target distribution before/after log transform | check whether log-scale modelling is justified |
| Predictor-target scatterplots | inspect relationships and nonlinearity |
| Prior predictive plots | validate priors before seeing the data |
| Trace plots | check MCMC mixing and convergence |
| Posterior density/HDI plots | interpret parameter uncertainty |
| Posterior predictive checks | verify whether simulated data resemble observed data |
| Grouped PPC by race category | check category-level calibration |
| LOO/WAIC comparison plots | compare predictive performance |
| Pareto-k plots | evaluate reliability of PSIS-LOO |

Typical generated figure patterns:

```text
fig*.png
fig_wt_*.png
fig_female_*.png
plot*.png
```

---

## Reproducibility Notes

- Some fitting notebooks may use a random subsample for computational feasibility because generated quantities can be memory-heavy for tens of thousands of observations.
- MCMC results may vary slightly across machines and package versions.
- Stan models may be recompiled automatically depending on the local CmdStan version.
- Notebooks should be run from the repository root to keep relative paths valid.
- If feature engineering in notebook `01` changes, all downstream notebooks should be rerun.
- For better long-term reproducibility, posterior objects should be saved as ArviZ `InferenceData` files, for example in NetCDF format.

---

## Limitations and Future Work

Main limitations:

- the analysis uses race-level aggregated data, not individual runner-level data;
- weather, technical terrain, aid-station structure and race-day conditions are not included;
- predictors such as distance, elevation gain and steepness are correlated, so individual coefficients require careful interpretation;
- current models do not include hierarchical effects for race category, event or country;
- some posterior inference steps use subsampling for computational reasons.

Recommended future extensions:

1. Add varying intercepts by `Race Category`.
2. Add varying intercepts by `Country` or event family.
3. Add nonlinear effects using splines or Gaussian processes.
4. Include weather and terrain technicality variables if available.
5. For female participation, include category-level effects explicitly.
6. Save all fitted models as ArviZ `InferenceData` objects.
7. Refactor the repository into `notebooks/`, `models/`, `data/`, `figures/` and `outputs/` folders.

---

## Final Project Conclusion

The project shows that Bayesian modelling is well suited for UTMB race-level analysis. Log-scale Student-t regression is the preferred approach for both mean finish time and winning time, while Beta-Binomial logistic regression is the preferred approach for female participation counts.

Across all tasks, the key statistical lesson is that the likelihood matters. Matching the likelihood to the data type — positive continuous time, heavy-tailed residuals or bounded counts — leads to more realistic uncertainty, better posterior predictive behaviour and stronger out-of-sample performance.