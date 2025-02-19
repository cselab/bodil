#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def upscale_grid_nodes(u, loc):
    dim = np.ndim(u)
    assert len(loc) == dim
    for l in loc:
        assert l in 'np'

    new_shape = []
    for s, l in zip(u.shape, loc):
        if l == 'n':
            new_shape.append(2 * s - 1)
        elif l == 'p':
            new_shape.append(2 * s)
    new_shape = tuple(new_shape)

    uf = np.zeros(new_shape, dtype=u.dtype)

    uf[tuple([slice(None, None, 2)] * dim)] = u

    for axis, l in enumerate(loc):
        if l == 'n':
            slices_l = tuple(slice(None, -2, 2) if i == axis else slice(None) for i in range(dim))
            slices_r = tuple(slice(2, None, 2) if i == axis else slice(None) for i in range(dim))
            slices_mid = tuple(slice(1, -1, 2) if i == axis else slice(None) for i in range(dim))
            uf[slices_mid] = (uf[slices_l] + uf[slices_r]) / 2
        elif l == 'p':
            slices_src = tuple(slice(None, -1, 2) if i == axis else slice(None) for i in range(dim))
            slices_dst = tuple(slice(1, None, 2) if i == axis else slice(None) for i in range(dim))
            uf[slices_dst] = (uf[slices_src] + np.roll(uf[slices_src], -1, axis=axis)) / 2
    return uf

def main():
    nx = 5
    ny = 7
    Lx = 2 * np.pi
    Ly = 1

    # periodic along x, non-periodic along y
    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=True)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = np.sin(X) + Y

    uf = upscale_grid_nodes(upscale_grid_nodes(u, 'pn'), 'pn')
    nxf, nyf = uf.shape
    xf = np.linspace(0, Lx, nxf, endpoint=False)
    yf = np.linspace(0, Ly, nyf, endpoint=True)

    fig, ax = plt.subplots()
    ax.plot(x, u[:,0], color='C0')
    ax.plot(xf, uf[:,0], 'o', color='C0')
    ax.plot(x, u[:,3], color='C1')
    ax.plot(xf, uf[:,3*4], 'o', color='C1')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u$')
    plt.show()

if __name__ == '__main__':
    main()
