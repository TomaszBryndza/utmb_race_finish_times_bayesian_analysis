// Model 2: Student-t Linear Regression for Race Finish Times (Robust)
// y ~ Student_t(nu, mu, sigma)
// mu = alpha + beta_dist * distance_std + beta_elev * elevation_std
// The Student-t likelihood is more robust to outlier races

data {
  int<lower=1> N;
  vector[N] y;              // finish time in hours
  vector[N] distance_std;   // standardized distance
  vector[N] elevation_std;  // standardized elevation gain
}

parameters {
  real alpha;               // intercept
  real beta_dist;           // effect of distance
  real beta_elev;           // effect of elevation gain
  real<lower=0.01> sigma;   // scale parameter (lower-bounded to avoid numerical issues)
  real<lower=1> nu;         // degrees of freedom (controls tail heaviness)
}

model {
  // Priors
  alpha ~ normal(10, 5);           // average finish time ~10h
  beta_dist ~ normal(5, 3);       // longer distance -> longer time
  beta_elev ~ normal(2, 2);       // more elevation -> longer time
  sigma ~ exponential(0.2);        // scale parameter
  nu ~ gamma(2, 0.1);             // degrees of freedom, allows heavy tails

  // Likelihood (Student-t for robustness against outliers)
  y ~ student_t(nu, alpha + beta_dist * distance_std + beta_elev * elevation_std, sigma);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  vector[N] mu;

  for (i in 1:N) {
    mu[i] = alpha + beta_dist * distance_std[i] + beta_elev * elevation_std[i];
    y_rep[i] = student_t_rng(nu, mu[i], sigma);
    log_lik[i] = student_t_lpdf(y[i] | nu, mu[i], sigma);
  }
}
