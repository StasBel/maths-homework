import numpy as np
import numpy.linalg as lng
from collections import namedtuple
from numerical.hw05 import TASKS
from optimization.hw06 import task2, Method, InputData, gen_Ab

Stat = namedtuple("Stat", ["num_iters", "num_ops"])


def simple_gradient(A, b, eps=1e-9):
    print("Starting simple gradient method with precision {}:".format(eps))
    m, n = A.shape
    b = A.T @ b
    A = A.T @ A
    x = np.random.rand(n)
    num_iter, num_ops = 0, 0
    while lng.norm(A @ x - b) > eps:
        num_iter += 1
        gr = A @ x - b
        alpha = (gr.T @ gr) / ((A @ gr).T @ gr)
        x -= alpha * gr
        num_ops += n ** 2 + 3 * n + 1
    assert np.allclose(A @ x, b)
    return Stat(num_iter, num_ops)


def arbitrary_gradient(A, b, eps=1e-9):
    print("Starting arbitrary gradient method with precision {}:".format(eps))
    m, n = A.shape
    b = A.T @ b
    A = A.T @ A
    x = np.random.rand(n)
    num_iter, num_ops = 0, 0
    # dirs = np.random.rand(n, n)
    dirs = np.eye(n)
    while lng.norm(A @ x - b) > eps:
        num_iter += 1
        for j in range(n):
            gr = A @ x - b
            di = dirs[:, j]
            alpha = (gr.T @ di) / ((A @ di).T @ di)
            x -= alpha * di
            num_ops += n ** 2 + 3 * n + 1
    assert np.allclose(A @ x, b)
    return Stat(num_iter, num_ops)


def conjugate_gradient(A, b, eps=1e-9):
    print("Starting conjugate gradient method with precision {}:".format(eps))
    m, n = A.shape
    b = A.T @ b
    A = A.T @ A
    x = np.random.rand(n)
    num_iter, num_ops = 0, 0
    r = b - A @ x
    p = r.copy()
    rs = r.T @ r
    while True:
        num_iter += 1
        Ap = A @ p
        alpha = rs / (p.T @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        if lng.norm(r) <= eps:
            break
        rs_new = r.T @ r
        p = r + (rs_new / rs) * p
        num_ops += 10 ** 2 + 5 * 10 + 2
        rs = rs_new
    assert np.allclose(A @ x, b)
    return Stat(num_iter, num_ops)


if __name__ == '__main__':
    A, b = gen_Ab()
    print(simple_gradient(A, b))
    print(arbitrary_gradient(A, b))
    print(conjugate_gradient(A, b))
