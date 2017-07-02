import numpy as np
import numpy.polynomial as pl
from itertools import chain
from collections import namedtuple
from functools import lru_cache
from bisect import bisect
import matplotlib.pyplot as plt
from numerical.hw07 import Stat, measure, sub_grid

Data = namedtuple("Data", ["degs", "x1", "y1", "dy1", "x2", "y2", "dy2"])


def read_data():
    with open("polynomials/p21.txt") as f:
        lines = f.readlines()
        n = int(lines[1].split()[3])
        degs = list(map(int, lines[2].split()[2:]))
        read_floats = lambda ln: list(map(float, lines[ln].split()))
        x1, y1, dy1 = read_floats(4), read_floats(6), read_floats(8)
        x2, y2, dy2 = read_floats(10), read_floats(12), read_floats(14)
    return Data(np.array(degs), np.array(x1), np.array(y1),
                np.array(dy1), np.array(x2), np.array(y2), np.array(dy2))


def ermit_polynom(x, y, dy):
    z = list(chain(*((sx,) * 2 for sx in x)))
    f1 = lambda i, j: dy[i // 2] if z[i] == z[j] else (y[i // 2 + 1] - y[i // 2]) / (z[j] - z[i])
    f = lambda i, j: f1(i, j) if j - i <= 1 else (f(i + 1, j) - f(i, j - 1)) / (z[j] - z[i])
    f = lru_cache(maxsize=None)(f)
    P = pl.Polynomial(0)
    for i in reversed(range(1, len(z))):
        P = (P + pl.Polynomial(f(0, i))) * pl.Polynomial([-z[i - 1], 1])
    P += pl.Polynomial(y[0])
    return P


def sweep_solve(A, f):
    n = len(f)
    a = lambda i: 0 if i <= 0 else A[i, i - 1]
    b = lambda i: A[i, i]
    c = lambda i: 0 if i >= n - 1 else A[i, i + 1]
    d = lambda i: f[i]
    cp = lambda i: c(i) / b(i) if i <= 0 else c(i) / (b(i) - a(i) * cp(i - 1))
    dp = lambda i: d(i) / b(i) if i <= 0 else (d(i) - a(i) * dp(i - 1)) / (b(i) - a(i) * cp(i - 1))
    x = np.empty(n)
    for i in reversed(range(n)):
        x[i] = dp(i) if i >= n - 1 else dp(i) - cp(i) * x[i + 1]
    return x


def ermit_spline(x, y, dy=None, output_d=False):
    n = len(x)
    Pk = lambda k, a, b, c, d: sum(h * pl.Polynomial([-x[k], 1]) ** i for h, i in zip((a, b, c, d), range(4)))
    ak = lambda k: y[k]
    bk = None
    hk = lambda k: x[k + 1] - x[k]
    fk = lambda k: (y[k + 1] - y[k]) / hk(k)
    if dy is not None:
        bk = lambda k: dy[k]
    else:
        alphak = lambda k: 1 / hk(k)
        bettak = lambda k: 2 * (1 / hk(k) + 1 / hk(k + 1))
        gammak = lambda k: 1 / hk(k + 1)
        deltak = lambda k: 3 * (fk(k + 1) / hk(k + 1) + fk(k) / hk(k))
        alpha = np.array(list(alphak(k) for k in range(1, n - 2)))
        betta = np.array(list(bettak(k) for k in range(n - 2)))
        gamma = np.array(list(gammak(k) for k in range(n - 3)))
        delta = np.array(list(deltak(k) for k in range(n - 2)))
        A = np.diag(alpha, -1) + np.diag(betta) + np.diag(gamma, 1)
        b = sweep_solve(A, delta)
        bk = lambda k: b[k - 1] if 1 <= k <= n - 2 else 0
    ck = lambda k: (3 * fk(k) - bk(k + 1) - 2 * bk(k)) / hk(k)
    dk = lambda k: (bk(k) + bk(k + 1) - 2 * fk(k)) / (hk(k) ** 2)

    @np.vectorize
    def P(sx):
        k = bisect(x, sx) - 1 if sx != x[-1] else len(x) - 2
        P = Pk(k, ak(k), bk(k), ck(k), dk(k))
        return P.deriv()(sx) if output_d else P(sx)

    return P


def task01(x, y, dy, degs):
    a, b = min(x), max(x)
    xg = np.linspace(a, b, num=1000)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    L = None
    for deg, axis in zip(degs, axes):
        axis.set_title("m={}".format(deg))
        print("Интерполяция для deg={}".format(deg))
        sx, sy, sdy = sub_grid(x, y, deg, dy)
        EP = ermit_polynom(sx, sy, sdy)
        ES = ermit_spline(sx, sy, sdy)
        # axis.plot(sx, sy, "ro")
        L = axis.plot(x, y, "b.", xg, EP(xg), "g-", xg, ES(xg), "r--")
        print("Число узлов, где значение Эрмитового полинома ближе к значению табличной функции: "
              "{} из {}".format(sum(abs(EP(a) - b) < abs(ES(a) - b) for a, b in zip(x, y)), len(x)))
        for name, stat in zip(["Polynom", "Spline"], [measure(x, y, EP), measure(x, y, ES)]):
            print("{}: \t {}".format(name, stat))
    fig.legend(handles=L, labels=["Points", "Ermit polynom", "Ermit spline"])
    plt.show()


IND = [0, 1, 3, 6, 9, 13, 14, 19, 21, 23, 24]
IND2 = [0, 1, 2, 4, 8, 11, 17, 20, 22, 23, 24]


def find_subgrid(x, y, dy):
    n = len(x)
    bind, bsx, bsy, bsdy, bP, bmae = None, None, None, None, None, float("inf")
    each_deg = 5000
    for deg in range(12, n // 2 + 1):
        for _ in range(each_deg):
            ind = sorted(np.random.choice(n - 2, size=deg - 3, replace=False) + 1)
            ind = np.hstack((0, ind, 24))
            sx, sy, sdy = x[ind], y[ind], dy[ind]
            P = ermit_polynom(sx, sy, sdy)
            stat = measure(x, y, P)
            if stat.mae < bmae:
                bind, bsx, bsy, bsdy, bP, bmae = ind, sx, sy, sdy, P, stat.mae
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


def task02(x, y, dy, bind):
    n = len(x)
    sx, sy, sdy = x[bind], y[bind], dy[bind]
    EP = ermit_polynom(sx, sy, sdy)
    ES = ermit_spline(sx, sy, sdy)
    print("Сетка: x={} \n y={} \n dy={}".format(sx, sy, sdy))
    print("Min mae: {}".format(measure(x, y, EP).mae))
    print("Polynom: \t {}".format(measure(x, y, EP)))
    print("Spline: \t {}".format(measure(x, y, ES)))
    # plot
    xg = np.linspace(min(x), max(x), num=1000)
    plt.plot(x, y, "r.")
    L = plt.plot(sx, sy, "bo", xg, EP(xg), "g-", xg, ES(xg), "r--")
    plt.legend(handles=L, labels=["Points", "Ermit polynom", "Ermit spline"])
    plt.show()


def task03(x, y, dy, deg):
    sx, sy, sdy = sub_grid(x, y, deg, dy)
    EP = ermit_polynom(sx, sy, sdy)
    ES = ermit_spline(sx, sy, sdy)
    ESw = ermit_spline(sx, sy)
    print("Funcs:")
    for name, P in zip(["Polynom", "Spline", "Spline w\o dy"], [EP, ES, ESw]):
        stat = measure(x, y, P)
        print("{}: \t {}".format(name, stat))

    # plot begin
    xg = np.linspace(min(x), max(x), num=1000)
    plt.plot(x, y, "r.")
    L = plt.plot(sx, sy, "bo", xg, ES(xg), "r--", xg, ESw(xg), "y-")
    plt.legend(handles=L, labels=["Points", "Ermit spline", "Ermit spline w/o dy"])
    plt.show()
    # plot end

    dEP = EP.deriv()
    dES = ermit_spline(sx, sy, sdy, output_d=True)
    dESw = ermit_spline(sx, sy, output_d=True)
    print("Deriatives:")
    for name, P in zip(["Polynom", "Spline", "Spline w\o dy"], [dEP, dES, dESw]):
        stat = measure(x, dy, P)
        print("{}: \t {}".format(name, stat))


if __name__ == '__main__':
    data = read_data()
    # task01(data.x1, data.y1, data.dy1, data.degs)
    # ind = find_subgrid(data.x1, data.y1, data.dy1)
    # print(measure(data.x1, data.y1, ermit_polynom(data.x1[ind], data.y1[ind], data.dy1[ind])).mae)
    # print(measure(data.x1, data.y1, ermit_polynom(data.x1[IND], data.y1[IND], data.dy1[IND])).mae)
    # task02(data.x1, data.y1, data.dy1, IND)
    # task03(data.x1, data.y1, data.dy1, data.degs[-1])
    # print("-" * 130)
    # task01(data.x2, data.y2, data.dy2, data.degs)
    # ind = find_subgrid(data.x2, data.y2, data.dy2)
    # print(ind)
    # print(measure(data.x2, data.y2, ermit_polynom(data.x2[ind], data.y2[ind], data.dy2[ind])).mae)
    # print(measure(data.x2, data.y2, ermit_polynom(data.x2[IND2], data.y2[IND2], data.dy2[IND2])).mae)
    # task02(data.x2, data.y2, data.dy2, IND2)
    task03(data.x2, data.y2, data.dy2, data.degs[-1])
