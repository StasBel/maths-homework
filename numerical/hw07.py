import math
import numpy as np
import numpy.polynomial as pl
import numpy.linalg as lng
import matplotlib.pyplot as plt
from operator import mul
from functools import reduce
from itertools import chain, combinations
from collections import namedtuple


def read_poly():
    with open("polynomials/p1.txt") as f:
        lines = f.readlines()
        n = int(lines[1].split()[3])
        degs = list(map(int, lines[2].split()[2:]))
        x = list(map(float, " ".join(lines[4:7]).split()))
        y = list(map(float, " ".join(lines[8:11]).split()))
    return np.array(x), np.array(y), np.array(degs)


def sub_grid(x, y, deg, dy=None):
    inds = list(map(int, np.linspace(0, len(x) - 1, num=deg + 1).tolist()))
    return (x[inds], y[inds]) if dy is None else (x[inds], y[inds], dy[inds])


def uniform_grid(a, b, n):
    return np.linspace(a, b, num=n)


def chebyshev_grid(a, b, n, *, inclusive=True, sort=True):
    n, xs = (n - 2, [a, b]) if inclusive else (n, [])
    xs.extend((a + b) / 2 + ((b - a) / 2) * math.cos((2 * k + 1) * math.pi / (2 * n + 2)) for k in range(n))
    xs = sorted(xs) if sort else xs
    return np.array(xs)


def lagrange_polynom(x, y):
    n = len(x)
    lij = lambda i, j: pl.Polynomial((-x[j] / (x[i] - x[j]), 1 / (x[i] - x[j])))
    li = lambda i: reduce(mul, (lij(i, j) for j in range(n) if i != j))
    P = sum(y * li(i) for i, y in enumerate(y))
    return P


def lsq_polynom(x, y, *, deg=None, w=None):
    n = len(x)
    deg = n - 1 if deg is None else deg
    w = np.ones(deg + 1) if w is None else w
    X = np.vander(x, N=deg + 1)[:, ::-1] * w
    a = lng.inv(X.T @ X) @ X.T @ y
    P = pl.Polynomial(a)
    return P


Stat = namedtuple("Stat", ["mae", "mre", "aae", "are"])


def measure(x, y, P):
    aes = list(abs(P(a) - b) for a, b in zip(x, y))
    mae = max(aes)
    aae = sum(aes) / len(aes)
    res = list(abs((P(a) - b) / P(a)) if P(a) != 0 else float("inf") for a, b in zip(x, y))
    mre = max(res)
    are = sum(res) / len(res)
    return Stat(mae, mre, aae, are)


def task01(x, y, degs):
    a, b = min(x), max(x)
    xg = np.linspace(a, b, num=1000)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    l1, l2, l3 = None, None, None
    for deg, axis in zip(degs, axes):
        axis.set_title("m={}".format(deg))
        print("Интерполяция для deg={}".format(deg))
        sx, sy = sub_grid(x, y, deg)
        LP = lagrange_polynom(sx, sy)
        SP = lsq_polynom(x, y, deg=deg)
        l1, l2, l3 = axis.plot(x, y, "b.", xg, LP(xg), "g-", xg, SP(xg), "r--")
        print("Число узлов, где значение интер. "
              "полинома ближе к значению табличной функции: "
              "{}".format(sum(abs(LP(a) - b) < abs(SP(a) - b) for a, b in zip(x, y))))
        for name, stat in zip(["Lagrange", "LSQ"], [measure(x, y, LP), measure(x, y, SP)]):
            print(stat)
            """
            print("Максимальная абсолютная погрешность "
                  "во всех узловых значениях заданной табличной "
                  "функции, {}: {}".format(name, stat.mae))
            print("Максимальная относительная погрешность "
                  "во всех узловых значениях заданной табличной "
                  "функции, {}: {}".format(name, stat.mre))
            print("Средняя абсолютная погрешность "
                  "во всех узловых значениях заданной табличной "
                  "функции, {}: {}".format(name, stat.aae))
            print("Средняя относительная погрешность "
                  "во всех узловых значениях заданной табличной "
                  "функции, {}: {}".format(name, stat.are))
            """
    fig.legend(handles=[l1, l2, l3], labels=["Points", "Lagrange", "LSQ"])
    plt.show()


IND = [0, 1, 2, 3, 6, 12, 14, 19, 20, 23, 24]


def find_subgrid(x, y):
    n = len(x)
    bind, bsx, bsy, bP, bmae = None, None, None, None, float("inf")
    each_deg = 30000
    for deg in range(12, n // 2 + 1):
        for _ in range(each_deg):
            ind = sorted(np.random.choice(n, size=deg - 1, replace=False))
            sx, sy = x[ind], y[ind]
            P = lagrange_polynom(sx, sy)
            stat = measure(x, y, P)
            if stat.mae < bmae:
                bind, bsx, bsy, bP, bmae = ind, sx, sy, P, stat.mae
        """
        for ind in combinations(range(n), deg - 1):
            ind = np.array(ind)
            sx, sy = x[ind], y[ind]
            P = lagrange_polynom(sx, sy)
            stat = measure(x, y, P)
            if stat.mae < bmae:
                bsx, bsy, bP, bmae = sx, sy, P, stat.mae
        """
    return bind


def task02(x, y, bind):
    n = len(x)
    sx, sy = x[bind], y[bind]
    bP = lagrange_polynom(sx, sy)
    bmae = measure(x, y, bP).mae
    bSP = lsq_polynom(x, y, deg=1)
    for deg in range(1, n // 2 + 1):
        P = lsq_polynom(x, y, deg=deg)
        stat = measure(x, y, P)
        if stat.mae < bmae:
            bSP = P
            print("Не лучше: LSQ степени {} с коэф. {} имеет "
                  "меньшую mae: {} < {}".format(deg, P, stat.mae, bmae))
            break
    else:
        print("Лучше: полином: {}, mae: {}".format(bP, min_mae))
    # plot
    xg = np.linspace(min(x), max(x), num=1000)
    plt.plot(x, y, "r.")
    l1, l2, l3 = plt.plot(sx, sy, "bo", xg, bP(xg), "g-", xg, bSP(xg), "r--")
    plt.legend(handles=[l1, l2, l3], labels=["Points", "Lagrange", "LSQ"])
    plt.show()
    return bP


def task03(x, y, P):
    deg = len(P)
    w = np.ones(deg + 1)
    bP = lsq_polynom(x, y, deg=deg, w=w)
    min_mae = measure(x, y, P).mae
    for i in range(0, deg + 1):
        for nw in np.linspace(-1, 1, num=1000):
            sw = w[i]
            w[i] = nw
            nP = lsq_polynom(x, y, deg=deg, w=w)
            stat = measure(x, y, nP)
            if stat.mae < min_mae:
                min_mae = stat.mae
            else:
                w[i] = sw
    bP = lsq_polynom(x, y, deg=deg, w=w)
    # print("Веса: {}".format(w))
    print("Статистика лучшего интер. полинома: {}".format(measure(x, y, P)))
    print("Статистика лучшего МНК полинома: {}".format(measure(x, y, bP)))
    xg = np.linspace(min(x), max(x), num=1000)
    l1, l2, l3 = plt.plot(x, y, "b.", xg, P(xg), "g-", xg, bP(xg), "r--")
    plt.legend(handles=[l1, l2, l3], labels=["Points", "Lagrange", "LSQ"])
    plt.show()


if __name__ == '__main__':
    x, y, degs = read_poly()
    # task01(x, y, degs)
    # ind = find_subgrid(x, y)
    P = task02(x, y, IND)
    task03(x, y, P)
