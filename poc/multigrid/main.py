#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def upscale_grid(u):
    dim = np.ndim(u)
    new_shape = tuple(2 * s for s in u.shape)
    uf = np.zeros(new_shape, dtype=u.dtype)

    # Copy original points
    uf[tuple([slice(None, None, 2)] * dim)] = u

    # Interpolate along each axis
    for axis in range(dim):
        slices1 = tuple(slice(None, -1, 2) if i == axis else slice(None) for i in range(dim))
        slices2 = tuple(slice(1, None, 2) if i == axis else slice(None) for i in range(dim))
        uf[slices2] = (uf[slices1] + np.roll(uf[slices1], shift=-1, axis=axis)) / 2

    # Remove edges.
    return uf[tuple([slice(0, -1)] * dim)]


def main():
    nx = 5
    L = 2 * np.pi

    x = np.linspace(0, L, nx)
    u = np.sin(x)

    # uf = interp_to_finer(u, depth=2)
    uf = upscale_grid(upscale_grid(u))
    xf = np.linspace(0, L, len(uf))

    fig, ax = plt.subplots()
    ax.plot(x, u)
    ax.plot(xf, uf, 'o')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u$')
    plt.show()

if __name__ == '__main__':
    main()
