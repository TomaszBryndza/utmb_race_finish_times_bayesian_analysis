// Model 4: Student-t Linear Regression for Winning Time (Robust)
// y ~ Student_t(nu, mu, sigma)
// mu = alpha + beta_dist * distance_std + beta_elev * elevation_std
// The Student-t likelihood is more robust to outlier races

data {
  int<lower=1> N;
  vector[N] y;              // winning time in hours
  vector[N] distance_std;   // standardized distance
  vector[N] elevation_std;  // standardized elevation gain
}

parameters {
  real alpha;               // intercept
  real beta_dist;           // effect of distance
  real beta_elev;           // effect of elevation gain
  real<lower=0.01> sigma;   // scale parameter
  real<lower=1> nu;         // degrees of freedom (controls tail heaviness)
}

model {
  // Priors
  alpha ~ normal(7, 4);            // average winning time ~7h
  beta_dist ~ normal(4, 2);       // longer distance -> longer winning time
  beta_elev ~ normal(1.5, 1.5);   // more elevation -> longer winning time
  sigma ~ exponential(0.3);        // scale parameter, mean ~3.3h
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
