# Title     : linear_regression
# Objective : prove that linear regression work
# Created by: belaevstanislav
# Created on: 11.05.17
library(foreign)
set.seed(1)

# read & scaling data
data <- data.matrix(read.dta("deaths.dta"))
data <- scale(data)

# data split
ratio <- 0.2
n <- nrow(data)
sample <- sample.int(n = nrow(data), size = floor(ratio * n), replace = F)
data_test <- data[sample,]
data_train <- data[- sample,]

# making wandermond matrix
make_wandermond <- function(x, p) {
    A <- 1
    for (k in 1 : p) {
        A <- cbind(A, x ^ k)
    }
    return(A)
}

# lr
poly_lr <- function(cx, cy, p, filename="lr_plot.png") {
    # calc w
    A <- make_wandermond(data_train[, cx], p)
    At <- t(A)
    y <- data_train[, cy]
    w <- solve(At %*% A) %*% At %*% y
    # mse and sse
    A <- make_wandermond(data_test[, cx], p)
    y <- data_test[, cy]
    y_pred <- A %*% w
    SSE <- sum((y_pred - y) ^ 2)
    MSE <- mean((y_pred - y) ^ 2)
    # plot
    png(filename)
    x <- data[, cx]
    y <- data[, cy]
    plot(x, y, xlab = cx, ylab = cy, col = "red", pch = 19)
    x <- seq(min(x), max(x), length = 1000)
    y <- make_wandermond(x, p) %*% w
    lines(x, y, col = "blue")
    dev.off()
    return(list("SSE" = SSE, "MSE" = MSE))
}

# Попробуем найти зависимость смертей от количества учетелей. Начнем с метода наименьших квадратов.
lr <- poly_lr(c("teachers"), c("deaths"), p = 1)
print(paste0("SSE: ", lr$SSE, " MSE: ", lr$MSE))
# Теперь попробуем найти полиномиальную зависимость с p=2, p=3 и p=4.
lr <- poly_lr(c("teachers"), c("deaths"), p = 2)
print(paste0("SSE: ", lr$SSE, " MSE: ", lr$MSE))
lr <- poly_lr(c("teachers"), c("deaths"), p = 3)
print(paste0("SSE: ", lr$SSE, " MSE: ", lr$MSE))
lr <- poly_lr(c("teachers"), c("deaths"), p = 4)
print(paste0("SSE: ", lr$SSE, " MSE: ", lr$MSE))
# Ошибка уменьшается. А вот с p=4 уже хуже, чем с p=3. Итого p=3 - оптимальное.
lr <- poly_lr(c("teachers"), c("deaths"), p = 3)
print(paste0("SSE: ", lr$SSE, " MSE: ", lr$MSE))
# Посмотрим на график "lr_plot.png". Получилось что-то похожее на правду. Итак, нашли зависимость.