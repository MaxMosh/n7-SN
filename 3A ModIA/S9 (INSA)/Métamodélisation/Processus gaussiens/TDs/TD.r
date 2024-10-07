A <- 7
B <- 0.1
a <- 1/2
b <- pi^4/5

ishigamiFun <- function(x){
  sin(x[, 1]) + 7 * sin(x[, 2])^2 + 0.1 * x[, 3]^4 * sin(x[, 1])
}

N <- 1000
xSim <- matrix(runif(3*N,min = -pi, max =  pi), ncol = 3) # matrice N x 3
ySim <- ishigamiFun(xSim) - A*a

i <- 1
#i <- 2
xi <- xSim[, i]
plot(xi, ySim)
t <- seq(from = min(xi), to = max(xi), length = 200)
lines(t, sin(t) * (1 + B * b), col = "red") # "+ A*a" d'après Qian
#lines(t, A*(sin(t)^2 - a), col = "red") # "+ A*a" d'après Qian



loessModel <- loess(y ~ x, data = data.frame(x = xi, y = ySim))
mainHat <- predict(loessModel, data.frame(x = t))
lines(t, mainHat, col = "blue",lwd = 4)
