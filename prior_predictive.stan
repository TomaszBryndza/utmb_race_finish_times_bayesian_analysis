// Prior predictive model for the log-time regression.
// No observed log_time and no likelihood: parameters are drawn directly
// from the priors in generated quantities, then propagated to predictions.
//
// The response variable being modelled is log(Mean Finish Time):
//   log(T_i) ~ student_t(nu, mu_i, sigma)
// Predictions are transformed back to hours via exp(), so finish times
// are guaranteed positive: time_rep[i] = exp(log_time_rep[i]).

data {
  int<lower=1> N;
  vector[N] distance_log_std;
  vector[N] elevation_log_std;
  vector[N] altitude_log_std;
  vector[N] steepness_std;
}

generated quantities {
  real alpha;
  real beta_dist;
  real beta_elev;
  real beta_alt;
  real beta_steep;
  real<lower=0> sigma;
  real<lower=2> nu;

  vector[N] mu;
  vector[N] log_time_rep;
  vector[N] time_rep;

  // Draw parameters directly from the priors (no data, no likelihood)
  alpha = normal_rng(log(15), 1);
  beta_dist = normal_rng(0.7, 0.4);
  beta_elev = normal_rng(0.2, 0.3);
  beta_alt = normal_rng(0.05, 0.1);
  beta_steep = normal_rng(0.15, 0.2);
  sigma = fabs(normal_rng(0, 0.35));   // half-normal on the log scale
  nu = 2 + gamma_rng(2, 0.1);          // nu_minus_two ~ Gamma(2, 0.1): mean ~20, little mass near 2

  for (i in 1:N) {
    mu[i] = alpha
            + beta_dist * distance_log_std[i]
            + beta_elev * elevation_log_std[i]
            + beta_alt * altitude_log_std[i]
            + beta_steep * steepness_std[i];

    log_time_rep[i] = student_t_rng(nu, mu[i], sigma);
    time_rep[i] = exp(log_time_rep[i]);   // back-transform to hours (always > 0)
  }
}
