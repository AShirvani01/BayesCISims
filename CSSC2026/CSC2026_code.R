

library(tidyverse)
library(rstanarm)
library(rstan)
library(BART)
library(gtools)       # for rdirichlet()
library(MCMCpack)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)



# 1. Simulate Data -------------------------------------------------------------

sas_origin <- as.Date("1960-01-01")

rhc <- read_csv("https://hbiostat.org/data/repo/rhc.csv") |>
  mutate(
    sadmdte  = as.Date(sadmdte,  origin = sas_origin),
    dthdte   = as.Date(dthdte,   origin = sas_origin),
    dschdte  = as.Date(dschdte,  origin = sas_origin),
    lstctdte = as.Date(lstctdte, origin = sas_origin),
    A        = as.integer(swang1 == "RHC"),
    Y_death  = as.integer(death == "Yes"),
    Y_los    = as.numeric(coalesce(dschdte, lstctdte) - sadmdte)
  )

print(rhc |> count(A, swang1))

# Confounders
covars <- c("age","sex","race","cat1",
            "meanbp1","hrt1","resp1","temp1","wtkilo1")


# 2. Bayesian G-computation ----------------------------------------------------

## 2.1: Parametric (Bayesian Logistic Regression) ------------------------------

fit_parametric <- stan_glm(
  Y_death ~ A + age + sex + race + cat1 + meanbp1 + hrt1 + resp1 + temp1 + wtkilo1,
  data = rhc,
  family = binomial(link = "logit"),
  prior = normal(0, 2.5), # eta_A, eta_L ~ N(0, 2.5)
  prior_intercept = normal(0, 5), # eta_0 ~ N(0,5)
  chains = 4,
  iter = 2000,
  warmup = 1000,
  seed = 42,
  refresh = 500
)

# Fit bayes Logistic Regression, sample eta draws from posterior

print(summary(fit_parametric, probs = c(0.025, 0.975)))


# --- Bayesian Bootstrap G-computation ---
M <- nrow(as.matrix(fit_parametric)) # number of posterior draws
n <- nrow(rhc)

# Counterfactual datasets
rhc_a1 <- rhc_a0 <- rhc
rhc_a1$A <- 1
rhc_a0$A <- 0

# Posterior predicted probabilities under each intervention
# mu_a1 and mu_a0 are M x n matrices
mu_a1_param <- posterior_epred(fit_parametric, newdata = rhc_a1)
mu_a0_param <- posterior_epred(fit_parametric, newdata = rhc_a0)

# Bayesian bootstrap: one fresh Dirichlet draw per MCMC iteration
set.seed(42)
psi_parametric <- numeric(M)
for(m in 1:M){
  bb_w <- as.vector(rdirichlet(1, rep(1, n)))
  psi_parametric[m] <- sum(bb_w * (mu_a1_param[m,] - mu_a0_param[m,]))
}

# Parametric Bayesian G-computation ATE (Risk Difference)
round(quantile(psi_parametric, c(0.025, 0.5, 0.975)), 4)
# P(ATE > 0 | data)
round(mean(psi_parametric > 0), 3)

# Plot posterior of Psi
p1 <- ggplot(data.frame(psi = psi_parametric), aes(x = psi)) +
  geom_density(fill = "steelblue", alpha = 0.4) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "red") +
  geom_vline(xintercept = median(psi_parametric),
             linetype = "solid", colour = "steelblue") +
  labs(x = expression(Psi ~ "(Risk Difference)"),
       y = "Posterior Density",
       title = "Parametric Bayes G-computation",
       subtitle = "Effect of RHC on 30-day Mortality") +
  theme_minimal(base_size = 13)

print(p1)



## 2.2 Non-parametric (BART) ---------------------------------------------------

X_train <- model.matrix(~ A + ., data = rhc[, c("A", covars)])[, -1]
Y_train <- rhc$Y_death

# Stack counterfactual test sets
X_a1 <- X_a0 <- X_train
X_a1[, "A"] <- 1
X_a0[, "A"] <- 0
X_test <- rbind(X_a1, X_a0)   # 2n rows

# Fit probit BART
set.seed(42)
bart_fit <- gbart(
  x.train = X_train,
  y.train = Y_train,
  x.test  = X_test,
  type    = "pbart",
  ndpost  = 1000,
  nskip   = 500)

# Extract posterior predictions (convert probit to probability)
mu_a1_bart <- t(pnorm(bart_fit$yhat.test[, 1:n]))          # n x M
mu_a0_bart <- t(pnorm(bart_fit$yhat.test[, (n+1):(2*n)])) # n x M

# Bayesian bootstrap standardization
bayes_boot <- function(mu_a1, mu_a0, seed = 42){
  set.seed(seed)
  n <- nrow(mu_a1); M <- ncol(mu_a1)
  sapply(1:M, function(m){
    w <- as.vector(rdirichlet(1, rep(1, n)))
    sum(w * (mu_a1[,m] - mu_a0[,m]))
  })
}

psi_bart <- bayes_boot(mu_a1_bart, mu_a0_bart)

# BART G-computation ATE (Risk Difference)
round(quantile(psi_bart, c(0.025, 0.5, 0.975)), 4)
# P(ATE > 0 | data) 
round(mean(psi_bart > 0), 3)



# 2.3 Bayesian PS Weighting -----------------------------------------------

# Fitting treatment model P(A|L) (bayesian logistic regression)
ps_fit <- stan_glm(
  A ~ age + sex + race + cat1 + meanbp1 + hrt1 + resp1 + temp1 + wtkilo1,
  data = rhc,
  family = binomial(link = "logit"),
  prior = normal(0, 2.5),
  prior_intercept = normal(0, 5),
  chains = 4,
  iter = 2000,
  warmup = 1000,
  seed = 42,
  refresh = 500
)

# Posterior draws of P(A=1|L): M x n matrix
ps_draws <- posterior_epred(ps_fit)
M_ps     <- nrow(ps_draws)
p_treat  <- mean(rhc$A)

# Posterior stabilized weights: M x n matrix
sw_draws <- ifelse(
  matrix(rhc$A, nrow = M_ps, ncol = n, byrow = TRUE) == 1,
  p_treat     / ps_draws,
  (1-p_treat) / (1 - ps_draws))

# Weighted outcome model per posterior weight draw
n <- nrow(rhc)
psi_ipw <- numeric(M)

for(m in 1:M){
  # Draw Bayesian bootstrap weights from Dir(1_n)
  bb_weights <- as.numeric(rdirichlet(1, rep(1, n)))
  
  # Combine BB weights with posterior IPW weights
  combined_weights <- bb_weights * sw_draws[m, ]
  combined_weights <- combined_weights / mean(combined_weights)  # normalize
  
  fit_m      <- lm(Y_death ~ A, data = rhc, weights = combined_weights)
  psi_ipw[m] <- coef(fit_m)["A"]
}

# Bayesian PS Weighting ATE (Risk Difference
round(quantile(psi_ipw, c(0.025, 0.5, 0.975)), 3)




# 3 Bayesian MSM ---------------------------------------------------------------

dat <- read_csv("CSSC2026/continuous_outcome_data.csv")
glimpse(dat)

library(bayesmsm)

# Step 1: Bayesian weight estimation
# Fitting conditional treatment models P(A_1 | W,L_1) and P(A_2 | W, L1, L2, A_1)
# Unstabilized weights (doesn't fit marginal treatment), left as 1
bayes_wt <- bayesweight(
  trtmodel.list = list(a_1 ~ w1 + w2 + L1_1 + L2_1, a_2 ~ w1 + w2 + L1_1 + L2_1 + L1_2 + L2_2 + a_1),
  data     = dat,
  n.iter   = 2500,
  n.burnin = 1500,
  n.thin = 5,
  parallel = TRUE,
  n.chains = 2,
  seed = 42
)

# Step 2: Bayesian MSM estimation
msm_fit <- bayesmsm(
  ymodel     = y ~ a_1 + a_2,
  family     = "gaussian",
  nvisit     = 2,
  reference  = c(0, 0),      # never treated
  comparator = c(1, 1),      # always treated
  data       = dat,
  wmean      = bayes_wt$weights,
  nboot      = 1000,
  optim_method = "BFGS",
  parallel   = TRUE,
  ncore = 4,
  seed       = 42)

summary_bayesmsm(msm_fit)
# Reference (no treatment, A = (0,0)): 2.317
# Comparator (all treatment, A = (1,1)): -0.836
# Relative Diff: -3.15

# Plot posterior ATE
plot_ATE(msm_fit,
         col_density = "steelblue",
         main = "Bayesian MSM: Posterior ATE\nAlways vs Never Treated (Continuous Outcome)")



# 4. Sensitivity Analysis -------------------------------------------------

sens_mod <- stan_model("CSSC2026/CSSC2026_sens.stan")

# Use RHC: standardize Y_death and age, omit a known confounder
stan_data_base <- list(
  N = nrow(rhc),
  Y = rhc$Y_death,                        # not rhc$Y
  A = rhc$A,
  L = as.vector(scale(rhc$age))           # not rhc$L_age
)

# Latent confounders for outcome and treatment, 16x2 grid
xi_grid <- expand.grid(
  xi1 = c(0, 0.5, 1.0, 1.5),    # U -> Y
  xi2 = c(0, 0.5, 1.0, 1.5)     # U -> A
)

# Running sensitivity analysis across grid

psi_grid <- vector("list", nrow(xi_grid)) #16,

for(k in 1:nrow(xi_grid)){ # 16 separate posteriors for psi
  
  xi1_k <- xi_grid$xi1[k]
  xi2_k <- xi_grid$xi2[k]
  
  cat(sprintf("Fitting model %d/%d: xi1=%.1f, xi2=%.1f\n",
              k, nrow(xi_grid), xi1_k, xi2_k))
  
  stan_data_k <- c(stan_data_base,
                   list(xi1 = xi1_k, xi2 = xi2_k))
  
  fit_k <- sampling(
    sens_mod,
    data    = stan_data_k,
    chains  = 2,
    iter    = 1500,
    warmup  = 500,
    seed    = 42,
    refresh = 0,    # suppress per-iteration output
    control = list(adapt_delta = 0.9,
                   max_treedepth = 10))
  
  # Check for divergences
  div <- sum(get_sampler_params(fit_k, inc_warmup = FALSE)[[1]][,"divergent__"])
  if(div > 0) cat(sprintf("  WARNING: %d divergent transitions\n", div))
  
  # Extract posterior draws of ATE (psi)
  psi_draws <- extract(fit_k, pars = "psi")[[1]]
  
  psi_grid[[k]] <- data.frame(
    xi1       = xi1_k,
    xi2       = xi2_k,
    xi1_label = paste0("xi1 = ", xi1_k,
                       " (OR=", round(exp(xi1_k), 2), ")"),
    xi2_label = paste0("xi2 = ", xi2_k,
                       " (OR=", round(exp(xi2_k), 2), ")"),
    mean      = mean(psi_draws),
    median    = median(psi_draws),
    lo        = quantile(psi_draws, 0.025),
    hi        = quantile(psi_draws, 0.975)
  )
  
  cat(sprintf("  ATE: %.4f [%.4f, %.4f]  P(ATE<0)=%.3f\n",
              mean(psi_draws),
              quantile(psi_draws, 0.025),
              quantile(psi_draws, 0.975)))
}

# Combine results
results <- bind_rows(psi_grid)



# 5. Bayesian Causal Forests (BCF) - CATE Estimation ---------------------------

library(bcf)

X <- model.matrix(~ ., data = rhc[, covars])[, -1]
Y_bcf <- rhc$Y_death
A_bcf <- rhc$A

# Step 1: Estimate propensity scores (logistic regression)
ps_glm <- glm(A_bcf ~ X, family = binomial)
ps_hat <- fitted(ps_glm)
print(summary(ps_hat))

# Step 2: Fit BCF
set.seed(42)
bcf_fit <- bcf(
  y          = Y_bcf,
  z          = A_bcf,
  x_control  = X,
  x_moderate = X,
  pihat      = ps_hat,
  nburn      = 500,
  nsim       = 1000)

# Step 3: Extract posterior draws
tau_draws    <- bcf_fit$tau
tau_hat      <- colMeans(tau_draws)               # was rowMeans - fix this
tau_lo       <- apply(tau_draws, 2, quantile, 0.025)
tau_hi       <- apply(tau_draws, 2, quantile, 0.975)
prob_benefit <- colMeans(tau_draws > 0)           # was rowMeans - fix this

# Population ATE from BCF
round(mean(tau_hat), 4)

# CATE by age subgroup
cate_df <- data.frame(
  tau_hat      = tau_hat,
  tau_lo       = tau_lo,
  tau_hi       = tau_hi,
  prob_benefit = prob_benefit,
  age          = rhc$age) |>
  mutate(age_q = cut(age,
                     quantile(age, 0:4/4),
                     labels = c("Q1 (youngest)","Q2","Q3","Q4 (oldest)"),
                     include.lowest = TRUE))

cate_sub <- cate_df |>
  group_by(age_q) |>
  summarise(
    mean_tau = mean(tau_hat),
    lo       = mean(tau_lo),
    hi       = mean(tau_hi),
    prob_ben = mean(prob_benefit),
    n        = n(),
    .groups  = "drop")

print(round(cate_sub[, -1], 4) |> cbind(age_q = cate_sub$age_q))

