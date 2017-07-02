import numpy as np


class Task1SubgradientDescent:
    def __init__(self, a, b):
        self.a, self.b = a, b

    def fit(self):
        x, iter_num = 0, 0
        while True:
            fs = self.a * x + self.b
            mf = fs.max()
            la, ra = self.a[fs == mf].min(), self.a[fs == mf].max()


if __name__ == '__main__':
    pass
