# Bayesian Analysis of UTMB Ultra-Trail Race Finish Times

This repository contains a Bayesian workflow for modelling ultra-trail race finish times using data from the UTMB World Series. The project focuses on explaining and predicting race times from course characteristics such as distance, elevation gain, steepness and altitude. It compares standard Gaussian regression models with robust Student-t alternatives and evaluates them using posterior diagnostics, posterior predictive checks, LOO-CV and WAIC.

The current branch, `unified_log_model`, uses a log-time formulation for the main mean finish-time models. This means the model is fitted to `log(Mean Finish Time)` and predictions are transformed back to hours with `exp(...)`, which guarantees strictly positive predicted finish times.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Research Questions](#research-questions)
- [Dataset](#dataset)
- [Bayesian Workflow](#bayesian-workflow)
- [Modelled Targets](#modelled-targets)
- [Model Specification](#model-specification)
- [Prior Design](#prior-design)
- [Posterior Analysis and Model Checking](#posterior-analysis-and-model-checking)
- [Model Comparison](#model-comparison)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [How to Run the Project](#how-to-run-the-project)
- [Reproducibility Notes](#reproducibility-notes)
- [Generated Figures](#generated-figures)
- [Limitations](#limitations)
- [Possible Extensions](#possible-extensions)

---

## Project Overview

Ultra-trail races differ substantially in distance, elevation gain, technical difficulty, altitude and location. As a result, finishing times are strongly right-skewed, strictly positive and affected by outlier races caused by terrain, weather, course design or data quality issues.

This project applies the full Bayesian workflow to model two race-level time targets:

1. **Mean Finish Time** — average finish time of participants in a race.
2. **Winning Time** — finish time of the race winner.

For each target, the project compares:

- a **Normal linear regression** model as a baseline;
- a **Student-t linear regression** model as a robust alternative.

The main modelling idea is that a Student-t likelihood can better handle heavy-tailed residuals and atypical races without forcing the whole model to inflate the residual scale parameter.

---

## Research Questions

The project addresses the following questions:

1. How do race distance, elevation-related variables and altitude influence expected ultra-trail finish times?
2. Can a Bayesian regression model quantify uncertainty around expected race times?
3. Does modelling finish time on the logarithmic scale improve the physical validity of predictions?
4. Does a robust Student-t likelihood provide better predictive performance than a Normal likelihood?
5. Are model conclusions consistent across mean participant performance and elite/winning performance?

---

## Dataset

The data comes from the Kaggle dataset:

```text
mgpoirot/utmb-world-race-daa
```

The raw dataset contains UTMB World Series race-level observations, including race identifiers, country, continent, race category, distance, elevation gain, participant counts and race timing variables.

The preprocessing notebook filters the data to the main UTMB race categories:

- `20K`
- `50K`
- `100K`
- `100M`

The cleaned dataset is saved as:

```text
utmb_processed.csv
```

Main raw variables used in the project include:

| Variable | Description |
|---|---|
| `Race Category` | UTMB race category: `20K`, `50K`, `100K`, `100M` |
| `Distance` | Race distance in kilometres |
| `Elevation Gain` | Total positive elevation gain in metres |
| `Mean Finish Time` | Mean finish time in decimal hours |
| `Winning Time` | Winning finish time in decimal hours |
| `N Participants` | Number of race participants |
| `Year` | Race year |
| `Country` | Country code/location |
| `Elevation` | Approximate altitude above sea level, when available |

Derived variables used for modelling include:

| Derived variable | Description |
|---|---|
| `log_time` | Natural logarithm of `Mean Finish Time` |
| `distance_log_std` | Standardized log-distance |
| `elevation_log_std` | Standardized elevation-related predictor used by the Stan model |
| `steepness_std` | Standardized log-transformed elevation gain per kilometre |
| `altitude_std` | Standardized altitude-related predictor |
| `distance_std` | Standardized distance, used in the winning-time models |
| `elevation_std` / `elevation_gain_std` | Standardized elevation-related variable used in the winning-time workflow |

> Note: the repository contains both legacy linear-scale standardized variables and the log-scale variables used by the current `unified_log_model` branch. The Stan files are the source of truth for the exact model input names.

---

## Bayesian Workflow

The project follows a structured Bayesian workflow:

1. **Problem formulation**
   - define the modelling objective;
   - identify target variables and predictors;
   - justify why finish time is suitable for probabilistic modelling.

2. **Exploratory data analysis**
   - inspect target distributions;
   - study skewness and heavy tails;
   - analyse relationships between distance, elevation gain and finish times;
   - motivate the use of log-transformation and robust likelihoods.

3. **Model specification**
   - define Normal and Student-t regression models;
   - specify predictors and likelihoods;
   - implement the models in Stan.

4. **Prior selection**
   - use weakly informative priors;
   - encode domain expectations about distance, elevation, steepness and altitude;
   - perform prior predictive checks.

5. **Posterior inference**
   - sample from the posterior using CmdStanPy;
   - inspect convergence diagnostics;
   - analyse posterior distributions of model parameters.

6. **Posterior predictive checking**
   - simulate replicated finish times;
   - compare observed and simulated distributions;
   - check summary statistics and behaviour by race category.

7. **Model comparison**
   - compare models using PSIS-LOO and WAIC;
   - inspect Pareto-k diagnostics;
   - evaluate robustness and predictive performance.

---

## Modelled Targets

### 1. Mean Finish Time

The primary workflow models:

```text
log_time = log(Mean Finish Time)
```

This is the main target for notebooks `01`–`05` and Stan models `model1_normal.stan` and `model2_student_t.stan`.

The log transformation is used because finish time is strictly positive and strongly right-skewed. Back-transforming posterior predictions with the exponential function ensures that predicted times in hours are always positive.

### 2. Winning Time

The additional workflow in `06_winning_time_bayesian_workflow.ipynb` models:

```text
Winning Time
```

This workflow compares two additional models:

- `model3_winning_normal.stan`
- `model4_winning_student_t.stan`

The winning-time models use standardized distance and elevation gain as predictors and compare Normal and Student-t likelihoods on the original hour scale.

---

## Model Specification

## Mean Finish Time Models

The main models are fitted to:

```text
y_i = log(T_i)
```

where `T_i` is the mean finish time in hours for race `i`.

The shared linear predictor is:

```text
mu_i = alpha
     + beta_dist  * distance_log_std_i
     + beta_elev  * elevation_log_std_i
     + beta_steep * steepness_std_i
     + beta_alt   * altitude_std_i
```

### Model 1: Normal Log-Time Regression

Implemented in:

```text
model1_normal.stan
```

Likelihood:

```text
log(T_i) ~ Normal(mu_i, sigma)
```

This model assumes light-tailed residuals on the log-time scale.

### Model 2: Student-t Log-Time Regression

Implemented in:

```text
model2_student_t.stan
```

Likelihood:

```text
log(T_i) ~ Student_t(nu, mu_i, sigma)
```

This model uses the same predictor set as Model 1 but replaces the Normal likelihood with a Student-t likelihood. The Student-t model is designed to be more robust to outlier races and heavy-tailed residuals.

In the current Stan implementation:

```text
nu = 2 + nu_minus_two
```

so the degrees of freedom parameter is constrained to be greater than 2.

---

## Winning Time Models

### Model 3: Normal Winning-Time Regression

Implemented in:

```text
model3_winning_normal.stan
```

Likelihood:

```text
Winning Time_i ~ Normal(mu_i, sigma)
```

with:

```text
mu_i = alpha + beta_dist * distance_std_i + beta_elev * elevation_std_i
```

### Model 4: Student-t Winning-Time Regression

Implemented in:

```text
model4_winning_student_t.stan
```

Likelihood:

```text
Winning Time_i ~ Student_t(nu, mu_i, sigma)
```

This model is the robust counterpart to Model 3.

---

## Prior Design

The project uses weakly informative priors that reflect domain knowledge while still allowing the data to dominate posterior inference.

For the mean finish-time models, the current Stan files use priors on the log-time scale. The regression coefficients have a multiplicative interpretation after exponentiation:

```text
exp(beta)
```

represents the approximate multiplicative change in finish time associated with a one-standard-deviation increase in the corresponding predictor.

The prior design reflects the following assumptions:

- longer races should generally take more time;
- higher elevation gain should generally increase race duration;
- steeper courses should generally be slower;
- altitude may have an additional physiological effect;
- the Student-t degrees-of-freedom parameter controls residual tail heaviness.

Prior predictive checks are performed in:

```text
02_model_specification_priors.ipynb
```

and the supporting Stan prior predictive model is provided in:

```text
prior_predictive.stan
```

---

## Posterior Analysis and Model Checking

Posterior analysis is split into separate notebooks:

| Notebook | Purpose |
|---|---|
| `03_posterior_model1_normal.ipynb` | Fit and evaluate the Normal log-time model |
| `04_posterior_model2_student_t.ipynb` | Fit and evaluate the Student-t log-time model |
| `06_winning_time_bayesian_workflow.ipynb` | Fit and evaluate the winning-time models |

The analysis includes:

- MCMC sampling with multiple chains;
- convergence checks using R-hat;
- effective sample size checks;
- divergence diagnostics;
- trace plots;
- marginal posterior plots;
- pair plots for parameter dependence;
- posterior predictive checks;
- comparison of observed and replicated summary statistics;
- category-level posterior predictive checks.

Generated posterior predictive quantities are stored in the Stan `generated quantities` blocks. For the log-time models, replicated values are produced on the log scale and then transformed back to hours:

```text
time_rep = exp(log_time_rep)
time_mu  = exp(mu)
```

---

## Model Comparison

Model comparison is performed in:

```text
05_model_comparison.ipynb
```

The compared models are:

| Model | Target scale | Likelihood | Robustness |
|---|---:|---|---|
| Model 1 | log-time | Normal | baseline |
| Model 2 | log-time | Student-t | robust |

The notebook compares predictive performance using:

| Criterion | Meaning |
|---|---|
| PSIS-LOO | Approximate leave-one-out cross-validation |
| WAIC | Widely Applicable Information Criterion |
| ELPD | Expected log predictive density; higher is better |
| Pareto-k | Reliability diagnostic for PSIS-LOO |

The winning-time workflow performs an analogous comparison for Model 3 and Model 4.

The exact numeric results are available in the executed notebooks and generated figures. In general, the comparison is designed to answer whether the Student-t likelihood gives better predictive performance and more reliable handling of outlier races than the Normal likelihood.

---

## Repository Structure

```text
.
├── 01_problem_formulation.ipynb
├── 02_model_specification_priors.ipynb
├── 03_posterior_model1_normal.ipynb
├── 04_posterior_model2_student_t.ipynb
├── 05_model_comparison.ipynb
├── 06_winning_time_bayesian_workflow.ipynb
├── data_exploration.ipynb
│
├── model1_normal.stan
├── model2_student_t.stan
├── model3_winning_normal.stan
├── model4_winning_student_t.stan
├── prior_predictive.stan
│
├── utmb_processed.csv
├── PROJECT_DESCRIPTION.md
├── notes.txt
├── todo.md
│
├── fig*.png
├── fig_wt_*.png
├── plot*.png
│
├── model1_normal
├── model1_normal.hpp
├── model2_student_t
├── model2_student_t.hpp
├── model3_winning_normal
├── model4_winning_student_t
└── prior_predictive
```

### Main notebooks

| File | Description |
|---|---|
| `01_problem_formulation.ipynb` | Data loading, cleaning, feature engineering and exploratory analysis |
| `02_model_specification_priors.ipynb` | Model definitions, prior rationale and prior predictive checks |
| `03_posterior_model1_normal.ipynb` | Posterior analysis of the Normal log-time model |
| `04_posterior_model2_student_t.ipynb` | Posterior analysis of the Student-t log-time model |
| `05_model_comparison.ipynb` | LOO, WAIC, Pareto-k diagnostics and final comparison |
| `06_winning_time_bayesian_workflow.ipynb` | Separate Bayesian workflow for winning time |
| `data_exploration.ipynb` | Additional exploratory analysis |

### Stan models

| File | Description |
|---|---|
| `model1_normal.stan` | Normal regression for log mean finish time |
| `model2_student_t.stan` | Student-t regression for log mean finish time |
| `model3_winning_normal.stan` | Normal regression for winning time |
| `model4_winning_student_t.stan` | Student-t regression for winning time |
| `prior_predictive.stan` | Prior predictive simulation model |

### Generated artifacts

The repository also contains generated figures, compiled Stan executables and generated `.hpp` files. These are outputs of the notebooks and Stan compilation process. They are useful for inspection but can be regenerated by rerunning the workflow.

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

### 3. Install Python dependencies

```bash
pip install numpy pandas scipy matplotlib arviz cmdstanpy kagglehub jupyter ipykernel
```

Optional, but useful for notebooks:

```bash
pip install ipywidgets tqdm
```

### 4. Install CmdStan

CmdStanPy requires a working CmdStan installation. After installing `cmdstanpy`, run:

```bash
python -m cmdstanpy.install_cmdstan
```

On Linux/macOS, make sure a C++ compiler and `make` are installed. On Windows, installing CmdStan may require a suitable C++ toolchain.

### 5. Register the environment as a Jupyter kernel

```bash
python -m ipykernel install --user --name utmb-bayes --display-name "Python (UTMB Bayes)"
```

---

## How to Run the Project

Run the notebooks from the repository root directory in the following order:

```text
01_problem_formulation.ipynb
02_model_specification_priors.ipynb
03_posterior_model1_normal.ipynb
04_posterior_model2_student_t.ipynb
05_model_comparison.ipynb
06_winning_time_bayesian_workflow.ipynb
```

Recommended workflow:

1. Open Jupyter:

```bash
jupyter notebook
```

or:

```bash
jupyter lab
```

2. Select the `Python (UTMB Bayes)` kernel.
3. Run `01_problem_formulation.ipynb` first to generate or refresh `utmb_processed.csv`.
4. Run `02_model_specification_priors.ipynb` to inspect prior assumptions.
5. Run posterior fitting notebooks `03` and `04`.
6. Run `05_model_comparison.ipynb` to compare the two mean finish-time models.
7. Run `06_winning_time_bayesian_workflow.ipynb` for the winning-time analysis.

---

## Reproducibility Notes

- The notebooks use random seeds where appropriate, but MCMC results may still vary slightly across machines and library versions.
- Some posterior fitting steps use a subsample of the full cleaned dataset to reduce memory usage and avoid kernel crashes.
- The full cleaned dataset is stored as `utmb_processed.csv`.
- The Stan executables may be recompiled automatically depending on your operating system and CmdStan version.
- Run notebooks from the repository root to ensure relative paths to `.stan` files, figures and CSV data work correctly.
- If you modify feature engineering in notebook `01`, rerun all downstream notebooks so that the Stan data and derived predictors remain consistent.

---

## Generated Figures

The repository includes generated figures from EDA, prior checks, posterior diagnostics, posterior predictive checks and model comparison.

Examples include:

| Figure pattern | Meaning |
|---|---|
| `fig01_*` to `fig23_*` | Main mean finish-time workflow figures |
| `fig_wt_*` | Winning-time workflow figures |
| `plot*.png` | Additional exploratory plots |

The most important figure groups are:

- target distributions;
- predictors vs targets;
- prior predictive simulations;
- trace plots;
- posterior predictive checks;
- posterior parameter distributions;
- LOO and WAIC comparisons;
- Pareto-k diagnostics;
- tail behaviour comparisons.

---

## Limitations

Important limitations of the current project:

1. **Race-level aggregation**  
   The dataset is modelled at race level, not individual runner level. Therefore, the model explains race-level average and winning times, not individual athlete performance.

2. **Limited course descriptors**  
   Distance and elevation gain are important, but they do not fully describe course difficulty. Trail surface, technicality, weather, aid stations and cutoff policies are not directly modelled.

3. **Potential feature-name ambiguity**  
   The repository contains both legacy and current standardized variables. The Stan files should be treated as the exact specification of the current models.

4. **Subsampling for computation**  
   Some posterior inference steps may use a subset of observations to reduce memory usage. This improves practicality but means results can depend slightly on the sampled subset.

5. **No hierarchical race structure**  
   The current models do not explicitly include hierarchical effects for country, year, race event or race category.

---

## Possible Extensions

Potential future improvements include:

- adding hierarchical effects for race category, year, country or event;
- modelling heteroscedasticity explicitly;
- including weather or terrain technicality data;
- using splines or Gaussian processes for nonlinear distance/elevation effects;
- modelling mean finish time and winning time jointly;
- comparing additional likelihoods such as lognormal, skew-normal or gamma models;
- using the full dataset with more memory-efficient generated quantities;
- adding scripts to reproduce all notebooks from the command line;
- moving notebooks, data, models and figures into separate directories for cleaner project structure.

---

## Technologies Used

The project uses:

- Python
- Jupyter Notebook
- Stan
- CmdStanPy
- ArviZ
- NumPy
- pandas
- SciPy
- Matplotlib
- kagglehub

---

## Project Status

The repository contains a complete Bayesian workflow for the current dataset and model family. The most important next step is repository cleanup: separating raw data, processed data, notebooks, Stan models and generated figures into dedicated folders, and adding a pinned `requirements.txt` or `environment.yml` for exact reproducibility.
