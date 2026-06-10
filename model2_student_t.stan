// Model 2: Student-t Linear Regression on LOG finish time (robust)
//
// log(T_i) ~ student_t(nu, mu_i, sigma)
// mu_i = alpha
//        + beta_dist  * distance_log_std
//        + beta_elev  * elevation_log_std
//        + beta_steep * steepness_std
//        + beta_alt   * altitude_std
//
// Identical predictor set to Model 1. Student-t likelihood is more
// robust to outlier races (extreme weather, data errors, atypical profiles).
//
// nu = 2 + nu_minus_two  => nu > 2 always (finite variance guaranteed).
// nu_minus_two ~ Gamma(2, 0.1): mean ~20, mode ~12, little mass near nu=2
// so prior-predictive tails are lighter than with Exponential(1/10).
//
// Prior rationale — see model1_normal.stan for detailed comments.
// Identical regression priors; only the likelihood and nu prior differ.

data {
  int<lower=1> N;
  vector[N] log_time;
  vector[N] distance_log_std;
  vector[N] elevation_log_std;
  vector[N] steepness_std;
  vector[N] altitude_std;
}

parameters {
  real alpha;
  real beta_dist;
  real beta_elev;
  real beta_steep;
  real beta_alt;
  real<lower=0> sigma;
  real<lower=0> nu_minus_two;
}

transformed parameters {
  real<lower=2> nu = 2 + nu_minus_two;
}

model {
  vector[N] mu = alpha
               + beta_dist  * distance_log_std
               + beta_elev  * elevation_log_std
               + beta_steep * steepness_std
               + beta_alt   * altitude_std;

  alpha        ~ normal(2.0,    1.0);
  beta_dist    ~ normal(0.7,  0.4);
  beta_elev    ~ normal(0.2,  0.3);
  beta_steep   ~ normal(0.15, 0.2);
  beta_alt     ~ normal(0.05, 0.1);
  sigma        ~ normal(0,    0.3);
  nu_minus_two ~ gamma(2, 0.1);

  log_time ~ student_t(nu, mu, sigma);
}

generated quantities {
  vector[N] mu;
  vector[N] log_time_rep;
  vector[N] time_rep;
  vector[N] time_mu;
  vector[N] log_lik;

  for (i in 1:N) {
    mu[i] = alpha
            + beta_dist  * distance_log_std[i]
            + beta_elev  * elevation_log_std[i]
            + beta_steep * steepness_std[i]
            + beta_alt   * altitude_std[i];

    log_time_rep[i] = student_t_rng(nu, mu[i], sigma);
    time_rep[i]     = exp(log_time_rep[i]);
    time_mu[i]      = exp(mu[i]);
    log_lik[i]      = student_t_lpdf(log_time[i] | nu, mu[i], sigma);
  }
}
