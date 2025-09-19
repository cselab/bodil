#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def f(x, y):
    return x**2 + 2 * y**2 + x * y

def grad_f_exact(x, y):
    return 2 * x + y, 4 * y + x

def eta_first_derivative_fs(x, y, axis, order=2):
    # Two-Dimensional First Derivatives, Full Space
    rsq = x**2 + y**2

    if order == 2:
        coeff = -2
    elif order == 4:
        coeff = -6 + 2 * rsq
    elif order == 6:
        coeff = -12 + rsq * (8 - rsq)
    elif order == 8:
        coeff = -20 + rsq * (20 - rsq * (5 - rsq/3))
    else:
        raise NotImplementedError(f'grad_pse: Not implemented for order {order}')

    xd = x if axis == 0 else y

    return coeff / np.pi * xd * np.exp(-rsq)

def eta_first_derivative_ls(x, y, axis, order=2):
    # Two-Dimensional First Derivatives, Left sided
    rsq = x**2 + y**2

    if order == 2:
        coeff = -4
    elif order == 4:
        coeff = -20 + 8 * rsq
    elif order == 6:
        coeff = -60 + rsq * (52 - 8 * rsq)
    elif order == 8:
        coeff = -140 + rsq * (196 - rsq * (64 - 16*rsq/3))
    else:
        raise NotImplementedError(f'grad_pse: Not implemented for order {order}')

    xd = x if axis == 0 else y

    return coeff / np.pi * xd * np.exp(-rsq)


def grad_pse(x, y, f, eps, order=2):
    dx = np.subtract.outer(x, x)
    dy = np.subtract.outer(y, y)
    fvals = f(x, y)
    df = np.subtract.outer(fvals, fvals)

    dfdx = -np.sum(df * eta_first_derivative_fs(dx / eps, dy / eps, axis=0, order=order) / eps**2, axis=0)
    dfdy = -np.sum(df * eta_first_derivative_fs(dx / eps, dy / eps, axis=1, order=order) / eps**2, axis=0)

    return dfdx, dfdy


def main():
    rng = np.random.default_rng(0xC0FFEE)
    n = 200
    L = 2
    x = rng.uniform(-L/2, L/2, n)
    y = rng.uniform(-L/2, L/2, n)

    rho = n / L**2
    eps = 2 / np.sqrt(rho)

    dfdx_exact, dfdy_exact = grad_f_exact(x, y)
    dfdx_pse, dfdy_pse = grad_pse(x, y, f, eps, order=2)

    fig, axes = plt.subplots(figsize=(8,4), ncols=2)
    ax = axes[0]
    ax.scatter(x, y, c=dfdx_exact)
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-L/2, L/2)
    ax.set_aspect('equal')

    ax = axes[1]
    ax.scatter(x, y, c=dfdx_pse)
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-L/2, L/2)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
