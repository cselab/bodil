#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def main():
    def bspline3(x):
        return np.where(x < 0,
                        0,
                        np.where(x < 1,
                                 x**2 / 2,
                                 np.where(x < 2,
                                          - x**2 + 3 * x - 1.5,
                                          np.where(x < 3,
                                                   (3 - x)**2 / 2,
                                                   0))))

    def basis(x):
        return bspline3((x * 3 + 1.5))

    L = 16
    x = np.linspace(0, L, 1024)

    sigma = 4
    n = int(L / sigma)
    nodes = np.linspace(0, L, n)


    fig, ax = plt.subplots()
    for i in range(5):
        coeffs = np.random.uniform(size=n)
        f = sum([ck * basis((x-xk) / (4*sigma)) for xk, ck in zip(nodes, coeffs)])
        ax.plot(x, f, c='C0')
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$f(x)$")
    ax.set_xlim(0, L)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
