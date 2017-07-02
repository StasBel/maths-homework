# Title     : to get task13
# Objective : ISLR, chapter: lr, task13
# Created by: belaevstanislav
# Created on: 12.05.17

# making A matrix for lr
make_A <- function(x, p) {
    A <- 1
    for (k in 1 : p) {
        A <- cbind(A, x ^ k)
    }
    return(A)
}

find_dep_lr <- function(x, y, p=1) {
    A <- make_A(x, p)
    At <- t(A)
    w <- solve(At %*% A) %*% At %*% y
    return(w)
}

# task13 subtask
task13 <- function(mean, sd, filename) {
    # a)
    x <- rnorm(100)
    # b)
    eps <- rnorm(100, mean = mean, sd = sd)
    # c)
    Y <- - 1 + 0.5 * x + eps
    # What is the length of the vector y?
    # Answer: 100
    # What are the values of β0 and β1 in this linear model?
    # Answer: β0 = -1, β1 = 0.5
    # d)
    png(filename)
    plot(x, Y, col = "red", pch = 19)
    # Comment on what you observe.
    # Answer: Четко прослеживается зависимость данных.
    # e)
    bt <- find_dep_lr(x, Y, p = 1)
    print(paste0("real b: [", - 1, ", ", 0.5, "] pred b: [", bt[1], ", ", bt[2], "]"))
    # Comment on the model obtained.
    # Answer: Все достаточно straitforward.
    # How do βˆ0 and βˆ1 compare to β0 and β1?
    # Answer: Очень похожи.
    # f)
    lx <- seq(min(x), max(x), length = 1000)
    ly <- make_A(lx, p = 1) %*% bt
    lines(lx, ly, col = "blue")
    # g)
    sqt = find_dep_lr(x, Y, p = 2)
    ly <- make_A(lx, p = 2) %*% sqt
    lines(lx, ly, col = "green")
    # legend
    legend("topleft", legend = c("points", "lq line", "p2 line"),
    bty = "n", col = c("red", "blue", "green"), pch = c(19, 19, 19))
    dev.off()
    MSE1 = mean((make_A(x, p = 1) %*% bt - Y) ^ 2)
    MSE2 = mean((make_A(x, p = 2) %*% sqt - Y) ^ 2)
    print(paste0("p=1 MSE:", MSE1, " p=2 MSE: ", MSE2))
    # Is there evidence that the quadratic term improves the model fit?
    # Answer: Yes.
}


# task13
set.seed(1)
task13(mean = 0, sd = 0.25, filename = "mean0sd025.png")
task13(mean = 0, sd = 0.15, filename = "mean0sd015.png")
task13(mean = 0, sd = 0.5, filename = "mean0sd05.png")