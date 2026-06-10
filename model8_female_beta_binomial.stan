// Model 7B: Beta-Binomial logistic regression for female participation counts
//
// Target:
//   n_women_i | n_participants_i, p_i, phi ~ Beta-Binomial(n_participants_i,
//                                                          p_i * phi,
//                                                          (1 - p_i) * phi)
//
// The linear predictor for p_i is identical to Model 7A. The extra parameter
// phi controls overdispersion. Large phi makes the model close to Binomial;
// smaller phi allows stronger race-to-race variability in female participation.
// This is the robust count-data analogue of replacing a Normal likelihood with
// a heavier-tailed likelihood in the time models.

data {
  int<lower=1> N;
  array[N] int<lower=0> n_women;
  array[N] int<lower=1> n_participants;

  vector[N] distance_log_std;
  vector[N] elevation_gain_log_std;
  vector[N] steepness_std;
  vector[N] altitude_std;
  vector[N] longitude_std;
  vector[N] latitude_std;
  vector[N] year_std;
}

parameters {
  real alpha;
  real beta_dist;
  real beta_elev_gain;
  real beta_steep;
  real beta_alt;
  real beta_lon;
  real beta_lat;
  real beta_year;
  real log_phi;
}

transformed parameters {
  real<lower=0> phi;
  vector[N] eta;
  vector[N] p;

  phi = exp(log_phi);

  eta = alpha
      + beta_dist      * distance_log_std
      + beta_elev_gain * elevation_gain_log_std
      + beta_steep     * steepness_std
      + beta_alt       * altitude_std
      + beta_lon       * longitude_std
      + beta_lat       * latitude_std
      + beta_year      * year_std;

  p = inv_logit(eta);
}

model {
  // Priors on the log-odds scale. Same regression priors as the Binomial model.
  alpha ~ normal(-1.0986122886681098, 1.0);

  beta_dist      ~ normal(-0.10, 0.35);
  beta_elev_gain ~ normal(-0.05, 0.35);
  beta_steep     ~ normal(-0.05, 0.35);

  beta_alt ~ normal(0.00, 0.25);
  beta_lon ~ normal(0.00, 0.30);
  beta_lat ~ normal(0.00, 0.30);

  beta_year ~ normal(0.20, 0.25);

  // Overdispersion prior.
  // phi is a concentration parameter. If phi is high, the latent race-level
  // probability is tightly concentrated around p_i and the model behaves like
  // Binomial. If phi is lower, the model allows more between-race variability.
  log_phi ~ normal(log(50), 1.0);

  // Likelihood.
  for (i in 1:N) {
    n_women[i] ~ beta_binomial(n_participants[i], p[i] * phi, (1 - p[i]) * phi);
  }
}

generated quantities {
  array[N] int n_women_rep;
  vector[N] p_mu;
  vector[N] prop_rep;
  vector[N] log_lik;

  for (i in 1:N) {
    p_mu[i] = p[i];
    n_women_rep[i] = beta_binomial_rng(n_participants[i], p[i] * phi, (1 - p[i]) * phi);
    prop_rep[i] = n_women_rep[i] * 1.0 / n_participants[i];
    log_lik[i] = beta_binomial_lpmf(n_women[i] | n_participants[i], p[i] * phi, (1 - p[i]) * phi);
  }
}
