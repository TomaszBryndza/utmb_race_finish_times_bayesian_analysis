// Model 2: Student-t Linear Regression on LOG finish time (robust)
// log(T_i) ~ student_t(nu, mu_i, sigma)
// mu_i = alpha + beta_dist * distance_log_std
//              + beta_elev * elevation_log_std
//              + beta_steep * steepness_std
//
// The Student-t likelihood on the log scale is robust to outlier races
// (technical terrain, extreme weather, data errors, atypical profiles).
// Modelling log_time keeps back-transformed predictions positive.

data {
  int<lower=1> N;
  vector[N] log_time;            // log(mean finish time [hours])
  vector[N] distance_log_std;    // standardized log-distance
  vector[N] elevation_log_std;   // standardized log-elevation
  vector[N] steepness_std;       // standardized steepness (elevation gain per km)
}

parameters {
  real alpha;
  real beta_dist;
  real beta_elev;
  real beta_steep;
  real<lower=0> sigma;            // scale on the log scale
  real<lower=0> nu_minus_two;     // reparameterization to enforce nu > 2
}

transformed parameters {
  real<lower=2> nu = 2 + nu_minus_two;   // finite-variance Student-t
}

model {
  vector[N] mu = alpha
               + beta_dist  * distance_log_std
               + beta_elev  * elevation_log_std
               + beta_steep * steepness_std;

  // Priors on the log-time scale
  alpha        ~ normal(log(10), 0.5);
  beta_dist    ~ normal(0.6, 0.3);
  beta_elev    ~ normal(0.2, 0.25);
  beta_steep   ~ normal(0.15, 0.25);
  sigma        ~ normal(0, 0.35);    // half-normal (sigma > 0)
  nu_minus_two ~ gamma(2, 0.1);      // mean ~ 20 (nu mean ~ 22), little mass near 2 -> lighter tails

  // Robust likelihood on the LOG scale
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
            + beta_steep * steepness_std[i];

    log_time_rep[i] = student_t_rng(nu, mu[i], sigma);
    time_rep[i]     = exp(log_time_rep[i]);   // always > 0
    time_mu[i]      = exp(mu[i]);
    log_lik[i]      = student_t_lpdf(log_time[i] | nu, mu[i], sigma);
  }
}
