// Prior predictive model — no observed log_time, no likelihood.
// Parameters are drawn directly from their priors inside generated quantities,
// then propagated to predictions. Run with fixed_param=True.
//
// Uses the same priors as model2_student_t.stan (the more general model).
// time_rep = exp(log_time_rep) is always strictly positive.
//
// NOTE: with fixed_param=True, CmdStanPy's stan_variable() may return zeros
// for scalar real variables. Use the Python simulation in notebook 02 instead
// (scipy / numpy replicating the same priors), which is fully equivalent.

data {
  int<lower=1> N;
  vector[N] distance_log_std;
  vector[N] elevation_log_std;
  vector[N] steepness_std;
  vector[N] altitude_std;
}

generated quantities {
  real alpha;
  real beta_dist;
  real beta_elev;
  real beta_steep;
  real beta_alt;
  real<lower=0> sigma;
  real<lower=2> nu;

  vector[N] mu;
  vector[N] log_time_rep;
  vector[N] time_rep;

  alpha      = normal_rng(0,    1.0);
  beta_dist  = normal_rng(0.7,  0.4);
  beta_elev  = normal_rng(0.2,  0.3);
  beta_steep = normal_rng(0.15, 0.2);
  beta_alt   = normal_rng(0.05, 0.1);
  sigma      = fabs(normal_rng(0, 0.3));
  nu         = 2 + gamma_rng(2, 0.1);

  for (i in 1:N) {
    mu[i] = alpha
            + beta_dist  * distance_log_std[i]
            + beta_elev  * elevation_log_std[i]
            + beta_steep * steepness_std[i]
            + beta_alt   * altitude_std[i];

    log_time_rep[i] = student_t_rng(nu, mu[i], sigma);
    time_rep[i]     = exp(log_time_rep[i]);
  }
}
