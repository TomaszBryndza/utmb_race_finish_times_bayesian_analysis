// Model 1: Normal Linear Regression on LOG finish time
// log(T_i) ~ Normal(mu_i, sigma)
// mu_i = alpha + beta_dist * distance_log_std + beta_elev * elevation_log_std
//
// We model log_time (not raw hours) so that back-transformed predictions
// time_rep = exp(log_time_rep) are guaranteed strictly positive.

data {
  int<lower=1> N;
  vector[N] log_time;            // log(mean finish time [hours])
  vector[N] distance_log_std;    // standardized log-distance
  vector[N] elevation_log_std;   // standardized log-elevation
}

parameters {
  real alpha;               // intercept on the log scale
  real beta_dist;           // effect of log-distance
  real beta_elev;           // effect of log-elevation
  real<lower=0> sigma;      // residual SD on the log scale
}

model {
  vector[N] mu = alpha
               + beta_dist * distance_log_std
               + beta_elev * elevation_log_std;

  // Priors on the log-time scale
  alpha     ~ normal(log(10), 0.5);   // typical race ~ 10 h
  beta_dist ~ normal(0.6, 0.3);       // exp(0.6) ~ 1.82x per +1 SD distance
  beta_elev ~ normal(0.2, 0.25);      // exp(0.2) ~ 1.22x per +1 SD elevation
  sigma     ~ normal(0, 0.35);        // half-normal (sigma > 0) on the log scale

  // Likelihood on the LOG scale
  log_time ~ normal(mu, sigma);
}

generated quantities {
  vector[N] mu;
  vector[N] log_time_rep;   // replicate on the log scale
  vector[N] time_rep;       // replicate back-transformed to hours
  vector[N] time_mu;        // exp(mu): median finish time in hours
  vector[N] log_lik;        // log-likelihood for log_time (for WAIC/LOO)

  for (i in 1:N) {
    mu[i] = alpha
            + beta_dist * distance_log_std[i]
            + beta_elev * elevation_log_std[i];

    log_time_rep[i] = normal_rng(mu[i], sigma);
    time_rep[i]     = exp(log_time_rep[i]);   // always > 0
    time_mu[i]      = exp(mu[i]);
    log_lik[i]      = normal_lpdf(log_time[i] | mu[i], sigma);
  }
}
