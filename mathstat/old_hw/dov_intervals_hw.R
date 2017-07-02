# Title     : dov intervals
# Objective : check if dov intervals theory work
# Created by: belaevstanislav
# Created on: 12.05.17
len1 <- function(n, eps=1e-2, a=0, sigma=1) {
    q <- qchisq(c(eps / 2, 1 - eps / 2), df = n)
    x <- rnorm(n, mean = a, sd = sigma)
    return(((q[2] - q[1]) / (q[1] * q[2])) * sum((x - a) ^ 2))
}

len2 <- function(n, eps=1e-2, a=0, sigma=1) {
    q <- qnorm(c(eps / 2, 1 - eps / 2))
    x <- rnorm(n, mean = a, sd = sigma)
    qs <- (q[2] - q[1]) / (q[1] * q[2])
    return(qs * sqrt(n) * (mean(x) - a))
}

plot_line <- function(x, y, filename) {
    png(filename)
    plot(x, y, col = "blue", type = "l")
    dev.off()
}

task1 <- function(m, k, s=1) {
    n <- as.integer(seq(1, m, length = k))
    l <- lapply(n, len1)
    plot_line(n, l, "len1_plot.png")
}

task2 <- function(m, k, s=1) {
    n <- as.integer(seq(1, m, length = k))
    l <- lapply(n, len2)
    plot_line(n, l, "len2_plot.png")
}

task1(100, 50)
task2(1000, 100)