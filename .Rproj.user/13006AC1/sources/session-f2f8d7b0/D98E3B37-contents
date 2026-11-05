
# 3.2 Partial Pooling of Conditional Causal Effects -----------------------

library(rstan)
library(latex2exp)
set.seed(1)


## Simulate data -----------------------------------------------------------

expit <- function(x) {
  1/(1+exp(-x))
}

n <- 500


W <- rnorm(n)

V <- numeric(n)
P_V <- c(3,3,2,1,1)/10 # marginal of V

for (i in 1:n) {
  V[i] <- sample(5, 1, prob=P_V)
}

gamma_V <- c(0, -0.5, 0., 0.5, -0.5)

# A | W,V
P_A <- expit(W + gamma_V)

A <- numeric(n)

for (i in 1:n) {
  A[i] <- rbinom(1, 1, prob=P_A[i])
}

eta_V <- c(-0.5, 0, 0.5, 0.6) #2:5

# Y | A,W,V ~ Ber(\mu(A, L)), L={W,V}
P_Y <- expit(-1 + W + (1 + outer(V, 2:5, "==")%*%eta_V)*A)

Y <- numeric(n)

for (i in 1:n) {
  Y[i] <- rbinom(1, 1, prob=P_Y[i])
}

## Bayesian Model ----------------------------------------------------------

V_mat <- model.matrix(~ as.factor(V))
n_v <- as.numeric(table(V))

stan_data <- list(
  n=n, 
  Pv=5, 
  W=W, 
  V=V_mat[order(V), , drop=F],
  A=A[order(V)],
  Y=Y[order(V)],
  n_v=n_v,
  ind=c(0, cumsum(n_v))
)

model <- stan_model(file="3-2/3-2_model.stan")

results <- sampling(model, data=stan_data, pars=c("odds_ratio", "overall_odds_ratio" ),
                    warmup=1000, iter=2000, chains=1, seed=1)

Psi_draws <- extract(results, pars='odds_ratio')[[1]]
overall_mean <- extract(results, pars='overall_odds_ratio')[[1]]


## Frequentist Model -------------------------------------------------------

vfac <- factor(V)

## estimate outcome model, which we'll then integrate over p(W)
freq_reg <- glm(Y ~  W + vfac + A + vfac*A, family=binomial('logit') )

## loop through strata of interest
Psi_freq <- numeric(5) ## shell to contain causal odds ratios
for(vf in 1:5){
  vval= as.factor(vf)
  
  ## predict probability under treatment and no treatment
  p1 = predict(freq_reg, data.frame(W, vfac=vval, A=1 ), type='response')
  p0 = predict(freq_reg, data.frame(W, vfac=vval, A=0 ), type='response') 
  
  ## standardize over empirical distribution Pv(W)
  marg_mean_y1 = mean( p1[ V==vf ]  )
  marg_mean_y0 = mean( p0[ V==vf ] )
  
  ## compute causal odds ratio
  Psi_freq[vf] = (marg_mean_y1/(1-marg_mean_y1)) / (marg_mean_y0/(1-marg_mean_y0))
}


# Plot --------------------------------------------------------------------

v_strata = ncol(V_mat)
post_mean = colMeans(Psi_draws)

png("ppooling_plot.png", width = 600, height = 500)

plot(post_mean, pch=20, col='blue', ylim=c(0,5),
     ylab=TeX("$\\Psi(v)$"), xlab=TeX("$V$"), axes=F )

axis_labs = paste0(v_strata, "\n(n=",table(V),")")

axis(1, at = 1:5, labels = axis_labs, padj = .5 )
axis(2, at = 0:5, labels = 0:5 )


### Plot posterior credible Intervals 
colfunc <- colorRampPalette(c("white", "lightblue"))
ci_perc = seq(.99,.01,-.01)
colvec = colfunc(length(ci_perc))
names(colvec) = ci_perc

for(i in ci_perc){
  pci = apply(Psi_draws, 2, quantile, probs=c( (1-i)/2, (1+i)/2  ) )
  segments(1:5,pci[1,], 1:5, pci[2,], col=colvec[as.character(i)], lwd=10 )
}
###

points(post_mean, col='steelblue', pch=20, lwd=8)
points(Psi_freq, col='black', pch=20, lwd=5)
abline(h= mean(overall_mean), col='steelblue', lty=2)

legend('topleft', 
       legend = c('Posterior Mean/Credible Band', 'MLE', "Overall Effect" ),
       col = c('steelblue', 'black', 'steelblue' ), pch=c(20,20,NA),
       lty = c(NA,NA,2), bty='n')

dev.off()
