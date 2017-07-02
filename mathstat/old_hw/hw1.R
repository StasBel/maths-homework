calc <- function(theta, n, N, maxk, Ti){
  # generate random sample
  s <- runif(n, min=0, max=theta)
  
  # k
  x <- seq(1, maxk, by = 0.1)
  
  # T plot
  y <- sapply(x, function(x) Ti(x, s))
  plot(x, y, cex=.1, col="blue", xlab="k", ylab="T")
  
  # convergence plot
  M <- function(k) {
    Ts = sapply(seq(1, N), function(x) Ti(k, runif(n, min=0, max=theta)))
    return(sum((Ts - theta) ^ 2) / N)
  }
  y <- sapply(x, function(x) M(x))
  plot(x, y, cex=.1, col="red", xlab="k", ylab="M")
  
}

# calculate m_k
m <- function(k, x){
  return(sum(x ^ k) / length(x))
}

# calculate T_1
T1 <- function(k, x){
  return(((k + 1) * m(k, x)) ^ (1/k))
}

# calculate T_2
T2 <- function(k, x){
  return((m(k, x) * gamma(1/(k + 1))) ^ (1/k))
}

# calculate T_3
T3 <- function(k, x){
  return((m(k, x)) ^ (1/k))
}

calc(theta = 1, n = 100, N = 100, maxk = 100, Ti = T1)