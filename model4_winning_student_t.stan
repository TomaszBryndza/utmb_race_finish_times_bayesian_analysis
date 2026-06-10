// Model 4: Student-t Linear Regression on LOG Winning Time (robust)
//
// log(WT_i) ~ Student_t(nu, mu_i, sigma)

data {
  int<lower=1> N;
  vector[N] log_winning_time;
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

  alpha        ~ normal(2.0, 1.0);
  beta_dist    ~ normal(0.7, 0.4);
  beta_elev    ~ normal(0.2, 0.3);
  beta_steep   ~ normal(0.15, 0.2);
  beta_alt     ~ normal(0.05, 0.1);
  sigma        ~ normal(0, 0.3);
  nu_minus_two ~ gamma(2, 0.1);

  log_winning_time ~ student_t(nu, mu, sigma);
}

generated quantities {
  vector[N] mu;
  vector[N] log_winning_time_rep;
  vector[N] winning_time_rep;
  vector[N] winning_time_mu;
  vector[N] log_lik;

  for (i in 1:N) {
    mu[i] = alpha
      + beta_dist  * distance_log_std[i]
      + beta_elev  * elevation_log_std[i]
      + beta_steep * steepness_std[i]
      + beta_alt   * altitude_std[i];

    log_winning_time_rep[i] = student_t_rng(nu, mu[i], sigma);
    winning_time_rep[i] = exp(log_winning_time_rep[i]);
    winning_time_mu[i] = exp(mu[i]);
    log_lik[i] = student_t_lpdf(log_winning_time[i] | nu, mu[i], sigma);
  }
}