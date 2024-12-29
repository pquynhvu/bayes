data{
   int N;                                 // # of obs
   int<lower=1> D;                        // # of communities
   int<lower=1> A;                        // # of age groups
   int<lower=0> K;                                     
   int<lower=0, upper=1> hospitalized[N]; 
   int<lower=0, upper=1> sex[N];          // indicator var
   int<lower=1> age[N];                   // factor levels of age groups
   int<lower=1> district[N];              // factor levels of district
   real<lower=0> admit_rate[N];              // factor levels of district
   int<lower=0, upper =1> y[N];           // outcome 
}
parameters {
  vector[K] beta;
  vector[A] alpha_age;
  vector[D] alpha_district;
  real<lower=0> sigma_age;
  real<lower=0> sigma_district;
}
transformed parameters  {
  real eta[N];
  for (i in 1:N) {
     eta[i] = beta[1] + beta[2]*hospitalized[i] + beta[3]*sex[i] + alpha_age[age[i]] + alpha_district[district[i]]*admit_rate[i];     
  }
}
model {
  // priors
  sigma_district ~ normal(0, 1);
  sigma_age ~ normal(0, 1);
  beta ~ normal(0, 1);
  alpha_district[1:D] ~ normal(0, sigma_district); 
  alpha_age[1] ~ normal(0, 1);
  alpha_age[2:A] ~ normal(alpha_age[1:(A-1)], sigma_age);
  // posterior
  y[1:N] ~ bernoulli_logit(eta[1:N]);
}
generated quantities {
  vector[N] y_rep;         // simulate data from the posterior          
  vector[N] log_lik;       // log-likelihood posterior 
  
  for (i in 1:N) {
    y_rep[i] = bernoulli_rng(inv_logit(eta[i]));
    log_lik[i] = bernoulli_logit_lpmf(y[i] | eta[i]);
  }
}
