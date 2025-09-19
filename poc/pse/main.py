#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import triangle as tr

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

    dfdx = -np.sum(df * eta_first_derivative_fs(dx / eps, dy / eps, axis=0, order=order), axis=0)
    dfdy = -np.sum(df * eta_first_derivative_fs(dx / eps, dy / eps, axis=1, order=order), axis=0)

    return dfdx, dfdy

def ring_segments(offset, n):
    idx = np.arange(offset, offset + n)
    return np.column_stack([idx, np.roll(idx, -1)])

def main():
    L = 2

    poly = {
        'vertices': np.array([[-L/2, -L/2],
                              [-L/2, +L/2],
                              [+L/2, +L/2],
                              [+L/2, -L/2],
                              ]),
        'segments': ring_segments(0, 4)
    }

    # Flags:
    #  q : quality mesh (default min angle ~20°; use q30 for 30°)
    #  a : max triangle area (e.g., a0.1)
    #  p : PSLG (respect segments)
    #  D : produce Delaunay (optional)
    area = 0.02
    mesh = tr.triangulate(poly, f"q30a{area}")
    vertices = mesh["vertices"]
    triangles = mesh["triangles"]
    x = vertices[:,0]
    y = vertices[:,1]

    n = len(x)
    rho = n / L**2
    eps = 2 / np.sqrt(rho)

    dfdx_exact, dfdy_exact = grad_f_exact(x, y)
    dfdx_pse, dfdy_pse = grad_pse(x, y, f, eps, order=2)

    norm = mcolors.Normalize(
        vmin=min(dfdx_exact.min(), dfdx_pse.min()),
        vmax=max(dfdx_exact.max(), dfdx_pse.max())
    )
    cmap = plt.cm.viridis

    fig, axes = plt.subplots(figsize=(8,4), ncols=2)
    ax = axes[0]
    ax.triplot(x, y, triangles, c='k', lw=0.1)
    ax.scatter(x, y, c=dfdx_exact, norm=norm, cmap=cmap)
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-L/2, L/2)
    ax.set_aspect('equal')
    ax.axis("off")

    ax = axes[1]
    ax.triplot(x, y, triangles, c='k', lw=0.1)
    ax.scatter(x, y, c=dfdx_pse, norm=norm, cmap=cmap)
    ax.set_xlim(-L/2, L/2)
    ax.set_ylim(-L/2, L/2)
    ax.set_aspect('equal')
    ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
