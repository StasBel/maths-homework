import numpy as np
import numpy.linalg as lng
import matplotlib.pyplot as plt
from random import random, randint, sample
from math import sqrt
from enum import Enum, auto
from abc import ABC, abstractmethod
from functools import reduce


def gen_Ab():
    m, n = np.random.randint(4, 10), np.random.randint(4, 10)
    A = np.random.uniform(-1, 1, size=(m, n))
    b = np.random.uniform(-1, 1, size=(m,))
    return A, b


def square_root_R(a, b, c):
    roots = np.roots([a, b, c]).tolist()
    l = list(filter(lambda x: not isinstance(x, complex), roots))
    return max(l) if l else None


def stack(matricies):
    return reduce(lambda a, y: np.vstack((a, y)),
                  (reduce(lambda a, x: np.hstack((a, x)), row) for row in matricies))


class ConjugateGradients:
    def __init__(self, A, b, c, *, eps=1e-7):
        self.A = A
        self.b = b
        self.c = c
        self.eps = eps

    def fit(self):
        f = lambda x: (1. / 2) * x.T @ self.A @ x - self.b.T @ x + self.c
        n = self.A.shape[0]
        x = np.random.rand(n)
        v = self.b - self.A @ x
        d = v.copy()
        vo = v.T @ v
        f_steps = [f(x)]
        for _ in range(n):
            Ad = self.A @ d
            alpha = vo / (d.T @ Ad)
            px = x.copy()
            x += alpha * d
            f_steps.append(f(x))
            v -= alpha * Ad
            vn = v.T @ v
            if lng.norm(vn) <= self.eps:
                break
            d = v + (vn / vo) * d
            vo = vn
        return x, f_steps


class AbstractGradientDescent(ABC):
    def __init__(self, n, f, df, *, eps=1e-7, max_iters=None):
        assert n > 0
        self.n = n
        self.f = f
        self.df = df
        assert eps > 0
        self.eps = eps
        assert max_iters is None or max_iters > 0
        self.max_iters = max_iters

    @abstractmethod
    def fit(self):
        pass


class SimpleGradientDescent(AbstractGradientDescent):
    def __init__(self, *args, **kwargs):
        lr = kwargs.pop("lr", None)
        super().__init__(*args, **kwargs)
        assert lr is not None and lr > 0
        self.lr = lr

    def fit(self):
        x = np.random.rand(self.n)
        fx = self.f(x)
        f_steps = [fx]
        while True:
            nx = x - self.lr * self.df(x)
            nfx = self.f(nx)
            f_steps.append(nfx)
            if lng.norm(nx - x) <= self.eps \
                    or lng.norm(nfx - fx) <= self.eps \
                    or (self.max_iters is not None and len(f_steps) > self.max_iters):
                x = nx
                break
            x, fx = nx, nfx
        return x, f_steps


class OptimalGradientDescent(AbstractGradientDescent):
    def __init__(self, *args, **kwargs):
        m = kwargs.pop("m")
        M = kwargs.pop("M")
        super().__init__(*args, **kwargs)
        self.m = m
        assert M != 0
        self.M = M

    def fit(self):
        x = np.random.rand(self.n)
        alpha = sqrt(self.m / self.M) if self.m else random()
        y = x
        fx = self.f(x)
        f_steps = [fx]
        while True:
            nx = y - (1. / self.M) * self.df(y)
            nfx = self.f(nx)
            f_steps.append(nfx)
            if lng.norm(nx - x) <= self.eps \
                    or lng.norm(nfx - fx) <= self.eps \
                    or (self.max_iters is not None and len(f_steps) > self.max_iters):
                x = nx
                break
            nalpha = square_root_R(1, alpha ** 2 - self.m / self.M, -(alpha ** 2))
            if not nalpha:
                raise ValueError("The iterative process broke down!")
            y = nx + ((alpha * (1 - alpha)) / (alpha ** 2 + nalpha)) * (nx - x)
            x, fx, alpha = nx, nfx, nalpha
        return x, f_steps


class CoordGradientDescent(AbstractGradientDescent):
    def __init__(self, *args, **kwargs):
        lr = kwargs.pop("lr", None)
        super().__init__(*args, **kwargs)
        assert lr is not None and lr > 0
        self.lr = lr

    def fit(self):
        x = np.random.rand(self.n)
        fx = self.f(x)
        f_steps = [fx]
        while True:
            nx = x.copy()
            for c in range(self.n):
                nx[c] -= self.lr * self.df(nx)[c]
            nfx = self.f(nx)
            f_steps.append(nfx)
            if lng.norm(nx - x) <= self.eps \
                    or lng.norm(nfx - fx) <= self.eps \
                    or (self.max_iters is not None and len(f_steps) > self.max_iters):
                x = nx
                break
            x, fx = nx, nfx
        return x, f_steps


class Method(Enum):
    CONJUGATE_GRADIENTS = auto()
    OPTIMAL_GRADIENT_DESCENT = auto()
    SIMPLE_GRADIENT_DESCENT = auto()
    COORD_GRADIENT_DESCENT = auto()
    ALL = auto()


def solve_task1(A, b, y, method, plt_steps=True):
    f = lambda x: ((x - y) ** 2).sum()
    x, steps = None, None
    if method == Method.CONJUGATE_GRADIENTS:
        m, n = A.shape
        At = (1. / 2) * A @ A.T
        bt = A @ y - b
        ct = 0
        cg = ConjugateGradients(At, bt, ct)
        x, steps = cg.fit()
        x = (-1. / 2) * A.T @ x
        x += y
    elif method == Method.SIMPLE_GRADIENT_DESCENT:
        m, n = A.shape
        aa = A @ A.T
        c = b - A @ y
        g = lambda x: (1. / 4) * x.T @ aa @ x + c.T @ x
        dg = lambda x: (1. / 2) * aa @ x + c
        eigvals = lng.eigvals((1. / 2) * aa)
        sm, M = abs(eigvals).min(), abs(eigvals).max()
        lr = 2. / (sm + M)
        sgd = SimpleGradientDescent(m, g, dg, lr=lr)
        x, steps = sgd.fit()
        x = (-1. / 2) * A.T @ x
        x += y
    elif method == Method.OPTIMAL_GRADIENT_DESCENT:
        m, n = A.shape
        aa = A @ A.T
        c = b - A @ y
        g = lambda x: (1. / 4) * x.T @ aa @ x + c.T @ x
        dg = lambda x: (1. / 2) * aa @ x + c
        eigvals = lng.eigvals((1. / 2) * aa)
        sm, M = abs(eigvals).min(), abs(eigvals).max()
        ogd = OptimalGradientDescent(m, g, dg, m=sm, M=M)
        x, steps = ogd.fit()
        x = (-1. / 2) * A.T @ x
        x += y
    elif method == Method.COORD_GRADIENT_DESCENT:
        m, n = A.shape
        aa = A @ A.T
        c = b - A @ y
        g = lambda x: (1. / 4) * x.T @ aa @ x + c.T @ x
        dg = lambda x: (1. / 2) * aa @ x + c
        eigvals = lng.eigvals((1. / 2) * aa)
        sm, M = abs(eigvals).min(), abs(eigvals).max()
        lr = 2. / (sm + M)
        sgd = CoordGradientDescent(m, g, dg, lr=lr)
        x, steps = sgd.fit()
        x = (-1. / 2) * A.T @ x
        x += y
    assert x is not None
    print("Minimun of function is {}, reached at x={}.".format(f(x), x))
    print("Number of steps: {}".format(len(steps)))
    if plt_steps:
        plt.plot(steps)
        plt.show()
        for step in steps:
            print(step)


def solve_task2(A, b, method, plt_steps=True):
    f = lambda x: ((A @ x - b) ** 2).sum()
    x, steps = None, None
    if method == Method.CONJUGATE_GRADIENTS:
        At = 2 * A.T @ A
        bt = 2 * b.T @ A
        ct = b.T @ b
        cg = ConjugateGradients(At, bt, ct)
        x, steps = cg.fit()
    elif method == Method.SIMPLE_GRADIENT_DESCENT:
        n = A.shape[1]
        df = lambda x: 2 * (A.T @ (A @ x - b))
        eigvals = lng.eigvals(2 * A.T @ A)
        m, M = abs(eigvals).min(), abs(eigvals).max()
        lr = 2. / (m + M)
        sgd = SimpleGradientDescent(n, f, df, lr=lr)
        x, steps = sgd.fit()
    elif method == Method.OPTIMAL_GRADIENT_DESCENT:
        n = A.shape[1]
        df = lambda x: 2 * (A.T @ (A @ x - b))
        eigvals = lng.eigvals(2 * A.T @ A)
        m, M = abs(eigvals).min(), abs(eigvals).max()
        ogd = OptimalGradientDescent(n, f, df, m=m, M=M)
        x, steps = ogd.fit()
    elif method == Method.COORD_GRADIENT_DESCENT:
        n = A.shape[1]
        df = lambda x: 2 * (A.T @ (A @ x - b))
        eigvals = lng.eigvals(2 * A.T @ A)
        m, M = abs(eigvals).min(), abs(eigvals).max()
        lr = 2. / (m + M)
        sgd = CoordGradientDescent(n, f, df, lr=lr)
        x, steps = sgd.fit()
    assert x is not None and steps is not None
    # print("Minimun of function is {}, reached at x={}.".format(f(x), x))
    print("Number of steps: {}".format(len(steps)))
    if plt_steps:
        plt.plot(steps)
        plt.show()
        for step in steps:
            print(step)


class InputData(Enum):
    AUTO_GENERATED = auto()


def create_solveable_slau(m, n):
    x = None
    while True:
        A = np.random.uniform(-10, 10, (m, n))
        b = np.random.uniform(-10, 10, m)
        try:
            x, *_ = lng.lstsq(A, b)
        except Exception:
            pass
        else:
            if lng.norm(x) > 1e-9:
                break
    return A, b


def task1(input_data, method, plt_steps):
    A, b, y = None, None, None
    if input_data == InputData.AUTO_GENERATED:
        n = randint(3, 10)
        m = n
        A, b = create_solveable_slau(m, n)
        y = np.random.uniform(-1, 1, n)
    assert A is not None and b is not None and y is not None
    if method == Method.ALL:
        for method in Method:
            if method != Method.ALL:
                print(method)
                solve_task1(A, b, y, method, plt_steps)
    else:
        solve_task1(A, b, y, method, plt_steps)


def task2(input_data, method, plt_steps):
    A, b = None, None
    if input_data == InputData.AUTO_GENERATED:
        # n, m = randint(3, 10), randint(3, 10)
        n, m = 10, 10
        A = np.random.uniform(-1, 1, (m, n))
        b = np.random.uniform(-1, 1, m)
    assert A is not None and b is not None
    if method == Method.ALL:
        for method in Method:
            if method != Method.ALL:
                print(method)
                solve_task2(A, b, method, plt_steps)
    else:
        solve_task2(A, b, method, plt_steps)


if __name__ == '__main__':
    task1(InputData.AUTO_GENERATED, Method.ALL, plt_steps=False)
    # task2(InputData.AUTO_GENERATED, Method.ALL, plt_steps=False)
