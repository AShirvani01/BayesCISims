
data {
  int<lower=0> N;
  array[N] int<lower=0, upper=1> Y;   // binary outcome
  array[N] int<lower=0, upper=1> A;   // binary treatment
  vector[N] L;                  // observed confounder (standardized)
  real xi1;                     // sensitivity param: U -> Y (log-OR)
  real xi2;                     // sensitivity param: U -> A (log-OR)
}
 
parameters {
  // Outcome model parameters
  real eta0;
  real eta1;
  real eta2;
 
  // Treatment model parameters
  real gam0;
  real gam1;
 
  // Latent unmeasured confounder - one value per subject
  // Sampled as missing data at every MCMC iteration
  vector[N] U;
}
 
model {
  // ---- Priors ----
  eta0 ~ normal(0, 3);
  eta1 ~ normal(0, 3);
  eta2 ~ normal(0, 3);
  gam0 ~ normal(0, 3);
  gam1 ~ normal(0, 3);
 
  // U ~ N(0,1): standardized latent confounder, independent of L
  // This is conservative: if U correlated with L, adjusting for L
  // would partially remove the confounding
  U ~ normal(0, 1);
 
  // ---- Likelihood ----
  // Both outcome and treatment depend on U - this is what makes U a confounder
  Y ~ bernoulli_logit(eta0 + eta1*to_vector(A) + eta2*L + xi1*U);
  A ~ bernoulli_logit(gam0 + gam1*L + xi2*U);
}
 
generated quantities {
  // G-computation: posterior ATE integrating over posterior U draws
  // For each MCMC draw, set A=1 and A=0 for all subjects,
  // keeping U fixed at its current posterior draw
  vector[N] mu1;
  vector[N] mu0;
  real psi;
 
  for(i in 1:N){
    mu1[i] = inv_logit(eta0 + eta1*1 + eta2*L[i] + xi1*U[i]);
    mu0[i] = inv_logit(eta0 + eta1*0 + eta2*L[i] + xi1*U[i]);
  }
  psi = mean(mu1) - mean(mu0);
}
