
# 3. Parametric Models in Point-treatment settings ------------------------



## Causal Dose Effects with AR1 Prior --------------------------------------

library(rstan)
library(latex2exp)
set.seed(1)


### Simulate data -----------------------------------------------------------

expit <- function(x) {
  1/(1+exp(-x))
}

K <- 10
n <- 100
D <- 0:(K-1)
D_mat <- matrix(D, nrow=n, ncol=K, byrow=TRUE) # n x K
L <- rnorm(n) # n,
L_mat <- matrix(L, nrow=n, ncol=K) # n x K
P_A <- expit(1 - (2/9)*D_mat + L_mat - 0.5*D_mat*L_mat) # n x K

# A | L
A <- numeric(n)

for (i in 1:n) {
  A[i] <- sample(D, size=1, prob=P_A[i,])
}

# Y | A,L ~ N(\mu(A,L), \phi=2)
Y <- rnorm(n, mean=5*pnorm(A - 5) - 5*L, sd=2) # n,



### Bayesian Model -------------------------------------------------------------------

A_mat <- model.matrix(Y ~ -1 + as.factor(A)) # n x K

tau <- rep(1, K-1)
tau[1] <- 10

stan_data <- list(n=n, num_A_levels=K-1, L=L, A=A_mat[,-1], Y=Y, mu=0, tau=tau)

model <- stan_model(file="3-1/3-1_model.stan")

results <- sampling(model, data=stan_data, warmup=500, iter=1000, chains=1, seed=1)

Psi_draws <- extract(results, pars='Psi')[[1]]


### Frequentist Model -------------------------------------------------------

freq_reg <- lm(Y ~ L + as.factor(A))
summary(freq_reg)

Psi_freq <- numeric(length = K-1)

Psi_freq[1] <- freq_reg$coefficients[3]
Psi_freq[2:(K-1)] <- freq_reg$coefficients[4:(K-1+2)] - freq_reg$coefficients[3:(K-2+2)]

### Plot --------------------------------------------------------------------

dose = 1:(K-1)
true_Psi = 5*pnorm(dose-5) - 5*pnorm((dose-1)-5)

png("dose_response_curve.png",width = 600, height = 500)
plot( dose, colMeans(Psi_draws), ylim=c(-4,4), 
      col='blue', pch=20, ylab=TeX("$\\Psi(k)$"), type='o',axes=F)

axis_labs = paste0(dose, " \n (n=",table(A[A!=0]),")")
axis(side = 1, seq(1:(K-1)), labels = axis_labs,tick = T, padj = .5)
axis(side = 2, seq(-4,4,1), labels = seq(-4,4,1), tick = T)

### Plot posterior credible Band 
colfunc <- colorRampPalette(c("white", "lightblue"))
ci_perc = seq(.99,.01,-.01)
colvec = colfunc(length(ci_perc))
names(colvec) = ci_perc

for(i in ci_perc){
  pci = apply(Psi_draws, 2, quantile, probs=c( (1-i)/2, (1+i)/2  ) )
  polygon(c(dose,rev(dose)),c(pci[1,],rev(pci[2,])),
          col = colvec[as.character(i)], border = FALSE)
}
###

lines(dose, colMeans(Psi_draws), col='steelblue', type='o', pch=20)

points(dose, Psi_freq, col='black', pch=20, type='o')
points(dose, true_Psi, col='red', pch=20, type='o' )
legend('bottomleft', 
       legend = c('Posterior Mean/Credible Band', 'MLE', 'True Curve'),
       col = c('steelblue', 'black', 'red'), pch=c(20,20,20), lty=c(1,1,1),
       bty='n')
dev.off()
