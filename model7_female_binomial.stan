// Model 7A: Binomial logistic regression for female participation counts
//
// Target:
//   n_women_i | n_participants_i, p_i ~ Binomial(n_participants_i, p_i)
//
// Linear predictor:
//   logit(p_i) = alpha
//              + beta_dist      * distance_log_std_i
//              + beta_elev_gain * elevation_gain_log_std_i
//              + beta_steep     * steepness_std_i
//              + beta_alt       * altitude_std_i
//              + beta_lon       * longitude_std_i
//              + beta_lat       * latitude_std_i
//              + beta_year      * year_std_i
//
// Why Binomial instead of Normal/Poisson?
//   The number of women is a bounded count: 0 <= n_women <= n_participants.
//   The Binomial likelihood models the count conditionally on the total field size.
//   This avoids the main bias of a simple Poisson model, where larger races would
//   automatically look like they have a stronger female-participation effect.

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
}

transformed parameters {
  vector[N] eta;
  vector[N] p;

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
  // Priors on the log-odds scale.
  // alpha centered at logit(0.25) = -1.0986: before seeing data, a typical race
  // is expected to have around 25% women, but the prior SD=1 is deliberately wide.
  alpha ~ normal(-1.0986122886681098, 1.0);

  // Course difficulty priors. Longer / harder / steeper races may have slightly
  // lower female share, but priors remain weak and allow positive effects.
  beta_dist      ~ normal(-0.10, 0.35);
  beta_elev_gain ~ normal(-0.05, 0.35);
  beta_steep     ~ normal(-0.05, 0.35);

  // Geographic priors. Weakly informative and centered close to no effect.
  beta_alt ~ normal(0.00, 0.25);
  beta_lon ~ normal(0.00, 0.30);
  beta_lat ~ normal(0.00, 0.30);

  // Year prior. Positive trend is plausible because female participation in
  // endurance sport may increase over time, but uncertainty is kept wide.
  beta_year ~ normal(0.20, 0.25);

  // Likelihood.
  for (i in 1:N) {
    n_women[i] ~ binomial_logit(n_participants[i], eta[i]);
  }
}

generated quantities {
  array[N] int n_women_rep;
  vector[N] p_mu;
  vector[N] prop_rep;
  vector[N] log_lik;

  for (i in 1:N) {
    p_mu[i] = p[i];
    n_women_rep[i] = binomial_rng(n_participants[i], p[i]);
    prop_rep[i] = n_women_rep[i] * 1.0 / n_participants[i];
    log_lik[i] = binomial_lpmf(n_women[i] | n_participants[i], p[i]);
  }
}
