import math
import random

TASK_1 = (None,
          lambda x: x ** 4 - 3 * x ** 2 + 75 * x - 10000,
          lambda x: 4 * x ** 3 - 6 * x + 75)
TASK_2 = (None,
          lambda x: (math.tan(x) - x) * math.cos(x),
          lambda x: x * math.sin(x))
TASK_3 = (None,
          lambda x: x ** 5 - x - 0.2,
          lambda x: 5 * x ** 4 - 1)
TASK_4 = lambda c: (None,
                    lambda x: x ** 2 - c,
                    lambda x: 2 * x)
TASK_5 = (3,
          lambda x: math.cos(x) + 1,
          lambda x: -math.sin(x))


class NewtonRoot:
    def __init__(self, task, eps=1e-9, itermax=100000):
        self.s, self.f, self.df = task
        self.eps = eps
        self.itermax = itermax

    def find(self):
        x, iternum = self.s or random.random(), 0
        while True:
            prev_x = x
            x -= (1. / self.df(x)) * self.f(x)
            if abs(x - prev_x) <= self.eps or iternum >= self.itermax:
                break
        assert abs(self.f(x) - .0) <= self.eps
        return x


if __name__ == '__main__':
    nr = NewtonRoot(TASK_4(5))
    print(nr.find())
    nr = NewtonRoot(TASK_5)
    print(nr.find())
