#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def main():
    """
    advection-diffusion equation, 1D, Dirichlet BC
    D lapl(u) - v grad(u) = 0
    u(0) = ul
    u(L) = ur
    """
    n = 128*32
    L = 1.0
    x = np.linspace(0, L, n, endpoint=True)

    v = 1.5
    D = 0.5

    lam = v / D

    ul = 1.5
    ur = 0.5
    A = (ul - ur) / (1 - np.exp(lam * L))
    C = ul - A
    uexact = A * np.exp(lam * x) + C

    u = uexact.copy()
    dx = x[1] - x[0]
    dudx = (np.roll(u, 1) - u)[1:-1] / dx
    d2udx2 = (2 * u - np.roll(u, 1) - np.roll(u, -1))[1:-1] / dx**2
    res = D * d2udx2 - v * dudx

    if 0:
        fig, ax = plt.subplots()
        ax.plot(x, uexact)
        ax.set_xlim(0, L)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$u$')
        plt.show()

    lim = np.max(np.abs(res))
    fig, ax = plt.subplots()
    ax.hist(res, bins=100, range=(-lim, lim), density=True)
    ax.set_xlabel(r'$\rho$')
    plt.show()

if __name__ == '__main__':
    main()
