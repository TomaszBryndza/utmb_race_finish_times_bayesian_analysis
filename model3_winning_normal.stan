// Model 3: Normal Linear Regression for Winning Time
// y ~ Normal(mu, sigma)
// mu = alpha + beta_dist * distance_std + beta_elev * elevation_std

data {
  int<lower=1> N;
  vector[N] y;              // winning time in hours
  vector[N] distance_std;   // standardized distance
  vector[N] elevation_std;  // standardized elevation gain
}

parameters {
  real alpha;               // intercept (winning time at average distance/elevation)
  real beta_dist;           // effect of distance
  real beta_elev;           // effect of elevation gain
  real<lower=0> sigma;      // residual standard deviation
}

model {
  // Priors
  alpha ~ normal(7, 4);            // average winning time ~7h, weakly informative
  beta_dist ~ normal(4, 2);       // longer distance -> longer winning time
  beta_elev ~ normal(1.5, 1.5);   // more elevation -> longer winning time
  sigma ~ exponential(0.3);        // residual SD, mean ~3.3h

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
