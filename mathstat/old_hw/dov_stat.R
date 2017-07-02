# Title     : dot stat
# Objective : check in dov stat work
# Created by: belaevstanislav
# Created on: 11.05.17
tetta <- 10
n <- 10
D <- 0
for (k in 1 : tetta) {
    D <- D + ((k ^ (n + 1) - (k - 1) ^ (n + 1)) ^ 2) / (k ^ n - (k - 1) ^ n)
}
D <- D / (tetta ^ n)
print(D)
print(tetta ^ 2 * (1 / n + 1))

E <- 0
for (k in 1: tetta) {
    E <- E + k * (k ^ n + (k - 1) ^ n)
}
E <- E / (tetta ^ n)
print(E)
print(tetta)