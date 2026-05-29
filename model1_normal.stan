// Model 1: Normal Linear Regression for Race Finish Times
// y ~ Normal(mu, sigma)
// mu = alpha + beta_dist * distance_std + beta_elev * elevation_std

data {
  int<lower=1> N;
  vector[N] y;              // finish time in hours
  vector[N] distance_std;   // standardized distance
  vector[N] elevation_std;  // standardized elevation gain
}

parameters {
  real alpha;               // intercept (mean finish time at average distance/elevation)
  real beta_dist;           // effect of distance
  real beta_elev;           // effect of elevation gain
  real<lower=0> sigma;      // residual standard deviation
}

model {
  // Priors
  alpha ~ normal(10, 5);           // average finish time ~10h, weakly informative
  beta_dist ~ normal(5, 3);       // longer distance -> longer time
  beta_elev ~ normal(2, 2);       // more elevation -> longer time
  sigma ~ exponential(0.2);        // residual SD, weakly informative

  // Likelihood
  y ~ normal(alpha + beta_dist * distance_std + beta_elev * elevation_std, sigma);
}

generated quantities {
  vector[N] y_rep;
  vector[N] log_lik;
  vector[N] mu;

  for (i in 1:N) {
    mu[i] = alpha + beta_dist * distance_std[i] + beta_elev * elevation_std[i];
    y_rep[i] = normal_rng(mu[i], sigma);
    log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
  }
}
