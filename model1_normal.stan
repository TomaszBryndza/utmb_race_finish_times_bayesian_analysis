// Model 1: Normal Linear Regression on LOG finish time
//
// log(T_i) ~ Normal(mu_i, sigma)
// mu_i = alpha
//        + beta_dist  * distance_log_std
//        + beta_elev  * elevation_log_std
//        + beta_steep * steepness_std
//        + beta_alt   * altitude_std
//
// All four predictors are standardized (mean 0, sd 1).
// Modelling log(T) guarantees time_rep = exp(log_time_rep) > 0.
//
// Prior rationale
// ---------------
// alpha  : Normal(0, 1)  — weakly informative intercept; NOT centred on the
//          sample mean. On the log-hour scale exp(-2,+2) = [0.14, 7.4] h,
//          intentionally vague so the data dominate.
//
// beta_dist : Normal(0.7, 0.4)  — Riegel's power-law (Riegel 1981) gives
//             time ∝ dist^1.06, so the log-log elasticity is ~1.06.
//             With SD(log dist) ≈ 0.6-0.8 in this dataset, the coefficient
//             on *standardised* log-distance is roughly 0.6-0.9 → prior mean 0.7.
//
// beta_elev : Normal(0.2, 0.3)  — positive from Minetti (2002): every extra
//             metre of ascent adds energy cost. Magnitude is smaller than
//             distance because elevation gain is partially captured by
//             steepness.
//
// beta_steep : Normal(0.15, 0.2)  — from Minetti / Strava GAP: at 5% average
//              grade (50 m/km) pace is ~30 % slower, at 10% (100 m/km) it is
//              ~2.4× slower. After log1p and standardisation, exp(0.15)≈1.16×
//              per SD is a plausible lower-bound effect.
//
// beta_alt  : Normal(0.05, 0.1)  — altitude hypoxia: above ~1500 m, VO2max
//             drops ~6-10 % per 1000 m (exercise physiology consensus).
//             With SD(altitude) ≈ 800 m, the expected coefficient per SD is
//             ~0.05-0.08 log-units. Weakly informative positive prior.
//
// sigma     : Normal(0, 0.3) truncated to sigma>0 (half-normal).
//             On the log scale, sigma≈0.25 means ±25 % multiplicative spread.

data {
  int<lower=1> N;
  vector[N] log_time;            // log(mean finish time [hours])
  vector[N] distance_log_std;    // standardised log-distance
  vector[N] elevation_log_std;   // standardised log-elevation gain
  vector[N] steepness_std;       // standardised log1p(elevation gain per km)
  vector[N] altitude_std;        // standardised altitude above sea level [m]
}

parameters {
  real alpha;
  real beta_dist;
  real beta_elev;
  real beta_steep;
  real beta_alt;
  real<lower=0> sigma;
}

model {
  vector[N] mu = alpha
               + beta_dist  * distance_log_std
               + beta_elev  * elevation_log_std
               + beta_steep * steepness_std
               + beta_alt   * altitude_std;

  alpha      ~ normal(0,    1.0);
  beta_dist  ~ normal(0.7,  0.4);
  beta_elev  ~ normal(0.2,  0.3);
  beta_steep ~ normal(0.15, 0.2);
  beta_alt   ~ normal(0.05, 0.1);
  sigma      ~ normal(0,    0.3);

  log_time ~ normal(mu, sigma);
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
            + beta_steep * steepness_std[i]
            + beta_alt   * altitude_std[i];

    log_time_rep[i] = normal_rng(mu[i], sigma);
    time_rep[i]     = exp(log_time_rep[i]);
    time_mu[i]      = exp(mu[i]);
    log_lik[i]      = normal_lpdf(log_time[i] | mu[i], sigma);
  }
}
