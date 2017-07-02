import math
import numpy as np
import numpy.linalg as lng
import numpy.random as rnd
import numpy.polynomial as poly
import scipy.linalg as slng
from collections import namedtuple
from numerical.hw05 import simple_iter_solve, relax_solve


def lu(A):
    return slng.lu(A, permute_l=True)


def cholesky(A):
    A = A.tolist()
    L = [[0.0] * len(A) for _ in range(len(A))]
    for i, (Ai, Li) in enumerate(zip(A, L)):
        for j, Lj in enumerate(L[:i + 1]):
            s = sum(Li[k] * Lj[k] for k in range(j))
            Li[j] = math.sqrt(Ai[i] - s) if (i == j) else (1.0 / Lj[j] * (Ai[j] - s))
    return np.array(L)


def lower_solve(L, b):
    n = b.shape[0]
    x = np.empty(n)
    for i in range(n):
        top = b[i] - (L[i, :i] * x[:i]).sum()
        bot = L[i, i]
        if not bot:
            if top:
                raise ValueError("Solution doesn't exist")
            else:
                x[i] = rnd.rand()
        else:
            x[i] = top / bot
    assert np.allclose(L @ x, b)
    return x


def upper_solve(L, b):
    n = b.shape[0]
    x = np.empty(n)
    for i in reversed(range(n)):
        top = b[i] - (L[i, i + 1:] * x[i + 1:]).sum()
        bot = L[i, i]
        if not bot:
            if top:
                raise ValueError("Solution doesn't exist")
            else:
                x[i] = rnd.rand()
        else:
            x[i] = top / bot
    assert np.allclose(L @ x, b)
    return x


def gauss_solve(A, b):
    L, U = lu(A)
    y = lower_solve(L, b)
    x = upper_solve(U, y)
    return x


Graph = namedtuple("Graph", ["n", "edges"])


def solve01(G):
    n, m = G.n, len(G.edges)
    A = np.zeros((n, m))
    for i, (a, b) in enumerate(G.edges):
        A[a - 1, i] += 1
        A[b - 1, i] -= 1
    b = np.zeros(m)
    A = np.vstack((A, np.ones(m)))
    b = np.hstack((np.zeros(n), 1))
    A, b = A.T @ A, A.T @ b
    # x = relax_solve(A, b)[0]  # number 1
    # x = simple_iter_solve(A, b)  # number 2, not allways working
    # x = gauss_solve(A, b)  # number 3
    x = abs(lng.lstsq(A, b)[0])  # correct np var
    return abs(np.round(lng.lstsq(A, b)[0], decimals=3))


def gen_data01():
    n = rnd.randint(3, 5 + 1)
    edges = []
    for _ in range(rnd.randint(3, int(n * (n - 1) / 2.) + 1)):
        edges.append((rnd.randint(1, n + 1), rnd.randint(1, n + 1)))
    return Graph(n, edges)


def print_ans01(w):
    print("Weights: {}".format(w))


def task01():
    G = gen_data01()
    # G = Graph(3, [(1, 2), (2, 3), (3, 1), (3, 1)])
    w = solve01(G)
    print_ans01(w)


def solve02(G):
    n = G.n
    A = np.zeros((n, n))
    for a, b in G.edges:
        A[a - 1, b - 1] += 1
    s = A.sum(axis=1)
    for i, s in enumerate(A.sum(axis=1)):
        if s == 0:
            A[i, i] = 1
        else:
            A[i] /= s
    A = A.T - np.eye(n)
    A = np.vstack((A, np.ones(n)))
    b = np.hstack((np.zeros(n), 1))
    A, b = A.T @ A, A.T @ b
    # x = relax_solve(A, b)[0]  # number 1
    # x = simple_iter_solve(A, b)  # number 2, not allways working
    x = gauss_solve(A, b)  # number 3
    # x = abs(lng.lstsq(A, b)[0])  # correct np var
    return abs(np.round(lng.lstsq(A, b)[0], decimals=3))


def gen_data02():
    n = rnd.randint(3, 5 + 1)
    edges = []
    for _ in range(rnd.randint(3, int(n * (n - 1) / 2.) + 1)):
        edges.append((rnd.randint(1, n + 1), rnd.randint(1, n + 1)))
    return Graph(n, edges)


def print_ans02(p):
    print("Probabilities: {}".format(p))


def task02():
    G = gen_data02()
    p = solve02(G)
    print_ans02(p)


def solve03(x, y, m):
    b = np.empty(m + 1)
    for k in range(m + 1):
        b[k] = (y * (x ** k)).sum()
    A = np.empty((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(m + 1):
            A[k, j] = (x ** (j + k)).sum()
    L = cholesky(A)
    y = lower_solve(L, b)
    w = upper_solve(L.T, y)
    return w


def gen_data03():
    n = 12
    x = np.linspace(10, 20, num=n)
    y = rnd.uniform(-1, 1, n)
    m = 5
    return x, y, m


def print_ans03(x, y, w):
    P = poly.Polynomial(w)
    print("Polynom: {}".format(P))
    error = sum((P(x) - y) ** 2 for x, y in zip(x, y))
    print("Error: {}".format(error))


def task03():
    # x, y, m = gen_data03()
    x = np.array([0, 1, 2, 3])
    y = np.array([3, 0, -1, 2])
    m = 2
    w = solve03(x, y, m)
    print_ans03(x, y, w)


def sub_task():
    A = np.random.rand(50, 50)
    A = A @ A.T
    L = cholesky(A)
    print(lng.norm(L @ L.T - A))


if __name__ == '__main__':
    # task01()
    # task02()
    task03()
    sub_task()
