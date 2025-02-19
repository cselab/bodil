#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def upscale_grid_nodes_periodic(u):
    dim = np.ndim(u)
    new_shape = tuple(2 * s for s in u.shape)
    uf = np.zeros(new_shape, dtype=u.dtype)
    uf[tuple([slice(None, None, 2)] * dim)] = u
    for axis in range(dim):
        slices_src = tuple(slice(None, -1, 2) if i == axis else slice(None) for i in range(dim))
        slices_dst = tuple(slice(1, None, 2) if i == axis else slice(None) for i in range(dim))
        uf[slices_dst] = (uf[slices_src] + np.roll(uf[slices_src], -1, axis=axis)) / 2
    return uf


def main():
    nx = 5
    L = 2 * np.pi

    x = np.linspace(0, L, nx, endpoint=False)
    u = np.sin(x)

    uf = upscale_grid_nodes_periodic(upscale_grid_nodes_periodic(u))
    xf = np.linspace(0, L, len(uf), endpoint=False)

    fig, ax = plt.subplots()
    ax.plot(x, u)
    ax.plot(xf, uf, 'o')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u$')
    ax.set_xlim(0, L)
    plt.show()

if __name__ == '__main__':
    main()
