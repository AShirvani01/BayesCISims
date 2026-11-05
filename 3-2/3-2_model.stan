
data {
  int<lower=0> n;
  int<lower=0> Pv; // Pv=5
  vector[n] W;
  matrix[n, Pv] V; // (1_n, ...)
  vector[n] A;
  array[n] int Y;
  array[Pv] int n_v;
  array[Pv+1] int ind;
}


parameters {
  vector[Pv] theta;
  real beta_W;
  vector[Pv] beta_V;
  real mu;
}


model {
  // priors
  beta_W ~ normal(0, 1);
  beta_V ~ normal(0, 1);
  // theta hyperparameter
  mu ~ normal(0, 1);
  
  theta[1] ~ normal(mu, 0.5);
  theta[2:Pv] ~ normal(mu - theta[1], 0.5);
  
  // likelihood
  for (i in 1:n) {
    Y[i] ~ bernoulli_logit(W[i]*beta_W + V[i]*beta_V + (V[i]*theta)*A[i]);
  }
}


generated quantities {

  real marg_mean_y1;
  real marg_mean_y0;
  real odds_1;
  real odds_0;

  vector[n] overall_cond_mean_y1;
  vector[n] overall_cond_mean_y0;
  vector[n] overall_bb_weights;
  real overall_odds_ratio;

  // cycle through strata of interest and compute causal Odds Ratio for each.
  vector[Pv] odds_ratio;

  for( v in 1:Pv ){

    // compute mean under strata v, under both A=1 and A=0
    int nv = n_v[v];
    int v_start = ind[v]+1;
    int v_end = ind[v+1];

    vector[nv] cond_mean_y1;
    vector[nv] cond_mean_y0;
    vector[nv] bb_weights;


    vector[nv] Wv = W[ v_start:v_end];
    matrix[nv, Pv] Vv = V[ v_start:v_end,];

    // compute conditional means.
    cond_mean_y1 = inv_logit( Wv*beta_W +Vv*beta_V + Vv*theta );
    cond_mean_y0 = inv_logit( Wv*beta_W + Vv*beta_V );

    // Bayesian bootstrap weights
    bb_weights = dirichlet_rng( rep_vector(1, nv) ) ;

    // taking average over bayesian bootstrap weights
    marg_mean_y1 = bb_weights' * cond_mean_y1;
    marg_mean_y0 = bb_weights' * cond_mean_y0;

    // compute odds ratio
    odds_1 = (marg_mean_y1/(1 - marg_mean_y1));
    odds_0 = (marg_mean_y0/(1 - marg_mean_y0));
    odds_ratio[v] = odds_1/odds_0;
  }

  for(i in 1:n){
    overall_cond_mean_y1[i] = inv_logit( W[i]*beta_W + V[i,]*beta_V + V[i, ]*theta );
    overall_cond_mean_y0[i] = inv_logit( W[i]*beta_W + V[i,]*beta_V );
  }

  overall_bb_weights = dirichlet_rng( rep_vector(1, n ) ) ;
  marg_mean_y1 = overall_bb_weights' * overall_cond_mean_y1 ;
  marg_mean_y0 = overall_bb_weights' * overall_cond_mean_y0 ;

  // compute odds ratio
  overall_odds_ratio = (marg_mean_y1/(1 - marg_mean_y1)) / (marg_mean_y0/(1 - marg_mean_y0));

}
