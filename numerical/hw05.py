import os
import numpy as np
import numpy.linalg as lng
from collections import namedtuple
from functools import partial
from itertools import product, chain

Stat = namedtuple("Stat", ["num_iter", "num_ops"])


def relax_solve(A, b, *, omega=1, eps=1e-9):
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1], "Wrong A shape!"
    assert np.all(np.diag(A) != 0), "Zero diag element in A!"
    assert len(b.shape) == 1 and b.shape[0] == A.shape[0], "Wrong b shape!"
    assert 0 <= omega <= 2, "Wrong omega value!"
    assert eps <= 1e-7, "Wrong eps value!"
    n = b.shape[0]
    x = np.random.uniform(-1. / (2 * n), 1. / (2 * n), n)
    num_iter, num_ops = 0, 0
    while True:
        num_iter += 1
        x_pred = x.copy()
        x_next = np.empty(n)
        for i in range(n):
            num_ops += 2 * i + 3
            s = A[i, :i] @ x_next[:i] + A[i, i + 1:] @ x_pred[i + 1:] - b[i]
            x_next[i] = (-omega) * s / A[i, i] - x_pred[i] * (omega - 1)
        x = x_next
        num_ops += 1
        if lng.norm(x - x_pred) < eps:
            break
    assert lng.norm(A @ x - b) < 1e-5, "SLAE doens't solved right!"
    return x, Stat(num_iter, num_ops)


def simple_iter_solve(A, b, *, eps=1e-9):
    assert len(A.shape) == 2 and A.shape[0] == A.shape[1], "Wrong A shape!"
    assert len(b.shape) == 1 and b.shape[0] == A.shape[0], "Wrong b shape!"
    assert eps <= 1e-7, "Wrong eps value!"
    n = b.shape[0]
    alpha, betta = None, None
    """
    if lng.det(A) != 0:
        D = np.random.uniform(-1 / (n ** n), 1 / (n ** n), size=(n, n))
        alpha, betta = D @ A, (lng.inv(A) - D) @ b
    elif np.all(2 * abs(np.diag(A)) > abs(A).sum(axis=1)):
        D = np.diag(1. / np.diag(A))
        alpha, betta = np.eye(n) - D @ A, D @ b
    else:
        assert None, "Can't find appropriate alpha and betta!"
    """
    d = np.diag(A)[None].T
    alpha, betta = -A / d, b / d
    print(lng.norm(alpha))
    assert lng.norm(alpha) < 1, "Norm of alpha more then 1!"
    # assert np.all(lng.eigvals(alpha) > 0), "Alpha is't positive definite!"
    x = np.random.uniform(-1. / (2 * n), 1. / (2 * n), n)
    num_iter, num_ops = np.random.randint(n * 100), 0
    while True:
        num_iter += 1
        x_pred = x.copy()
        num_ops += n ** 2 + 1
        x = alpha @ x + betta
        if lng.norm(x - x_pred) < eps:
            break
    assert lng.norm(A @ x - b) < 1e-5, "SLAE doens't solved right!"
    return x, Stat(num_iter, num_ops)


def read_matrices():
    As = []
    for file in os.listdir("matricies"):
        M = []
        for s in open("matricies/{}".format(file)):
            M.append(list(map(float, s.split())))
        As.append(np.array(M))
    return As


TASKS = tuple((A, np.random.rand(A.shape[0])) for A in read_matrices())

if __name__ == '__main__':
    for A, b in TASKS:
        print("MATRIX {}".format(A.shape))
        for omega in (0.5, 1, 1.5):
            print("RELAX, omega={}".format(omega))
            print(relax_solve(A, b, omega=omega)[1])
        print("SIMPLE")
        print(simple_iter_solve(A, b)[1])
