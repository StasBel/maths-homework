# Title     : task14
# Objective : ISLR, chapter: lr, task14
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

# full solve lr task
solve_lr <- function(x, y, p = 1) {
    A <- make_A(x, p)
    At <- t(A)
    w <- solve(At %*% A) %*% At %*% y
    MSE <- mean((A %*% w - y) ^ 2)
    return(list("w" = w, "MSE" = MSE))
}

# task14 subtask
task14 <- function(x1, x2, y, w) {
    # b)
    png("x12scatterplot.png")
    plot(x1, x2, col = "red", pch = 19)
    # What is the correlation between x1 and x2?
    # Answer: Зависимость явно есть.
    # c)
    x = cbind(x1, x2)
    lqs = solve_lr(x, y, p = 1)
    print(cbind(lqs$w, w))
    # Can you reject the null hypothesis H0 : β1 = 0?
    # Answer: Yes.
    # How about the null hypothesis H0 : β2 = 0?
    # Answer: Yes.
    # d)
    lqs_onlyx1 = solve_lr(x1, y, p = 1)
    print(cbind(lqs_onlyx1$w, w))
    # Can you reject the null hypothesis H0 : β1 = 0?
    # Answer: No.
    # e)
    lqs_onlyx2 = solve_lr(x2, y, p = 1)
    print(cbind(lqs_onlyx2$w, w))
    # Can you reject the null hypothesis H0 : β1 = 0?
    # Answer: Yes.
    # f)
    # Do the results obtained in (c)–(e) contradict each other? Explain your answer.
    # Answer: No. Yes. In a way, yes. Maybe.
}

# task14
# a)
set.seed(1)
options(warn = - 1)
x1 = runif(100)
x2 = 0.5 * x1 + rnorm(100) / 10
y = 2 + 2 * x1 + 0.3 * x2 + rnorm(100) # <=> 2 + 2.15 * x_1 + (0.03 rnorm(100) + rnorm(100))
w = c(2, 2, 0.3)
# Write out the form of the linear model. What are the regression coefficients?
# Answer: b0 = 2, b1 = 2, b2 = 0.3, eps ~ N(0, 1)
# Real answer: b0 = 2, b1 = 2.15, eps ~ 0.03 * N(0, 1) + N(0, 1) ~ N(0, 1.0009)
task14(x1, x2, y, w)
x1 = c(x1, 0.1)
x2 = c(x2, 0.8)
y = c(y, 6)
task14(x1, x2, y, w)
# What effect does this new observation have on the each of the models?
# Answer: Строго негативный.