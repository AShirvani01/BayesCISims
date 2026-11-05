data {
  int<lower=0> N;
  vector[N] L;
  vector[N] A;
  vector[N] Y;
}


parameters {
  real theta;
  real beta;
  real delta1;
  real phi; // uniform prior
  real delta;
}


model {
  // priors
  theta ~ normal(0, 3);
  beta ~ normal(0, 3);
  delta1 ~ normal(0, 1/sqrt(3));
  // delta ~ normal(0, 3^(-1/2));
  delta ~ gamma(1, 3);
  
  // likelihood
  Y ~ normal(theta*A + beta*L, phi);
}


generated quantities {
  real psi1;
  real psi2;
  real psi3;
  
  psi1 = theta + delta;
  psi2 = theta - delta;
  psi3 = theta + delta1;
}
