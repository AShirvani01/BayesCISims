
data {
  int<lower=0> n;
  int num_A_levels;
  vector[n] L;
  matrix[n, num_A_levels] A;
  vector[n] Y;
  // theta prior hyperparameters
  real mu; // set to 0
  vector<lower=0>[num_A_levels] tau; // tau[1]=10, 1 otherwise
}


parameters {
  vector[num_A_levels] theta;
  real beta;
  real<lower=0> phi;
}


model {
  // priors
  theta[1] ~ normal(mu, tau[1]);
  theta[2] ~ normal(2*theta[1], tau[2]);
  for (k in 3:num_A_levels) {
    theta[k] ~ normal(2*theta[k-1] - theta[k-2], tau[k]);
  }
  // nuisance priors
  beta ~ normal(0, 10); // linear coefficient for L
  phi ~ cauchy(0,10); // variance of likelihood
  
  // likelihood
  Y ~ normal(L*beta + A*theta, phi);
}


generated quantities {
  vector[num_A_levels] Psi;
  
  Psi[1] = theta[1];
  for(k in 2:num_A_levels){
    Psi[k] = theta[k] - theta[k-1];
  }
}
