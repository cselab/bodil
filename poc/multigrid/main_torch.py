#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch

def upscale_grid_nodes(u):
    dim = np.ndim(u)
    new_shape = tuple(2 * s - 1 for s in u.shape)
    uf = torch.zeros(new_shape, dtype=u.dtype)
    uf[tuple([slice(None, None, 2)] * dim)] = u
    for axis in range(dim):
        slices_l = tuple(slice(None, -2, 2) if i == axis else slice(None) for i in range(dim))
        slices_r = tuple(slice(2, None, 2) if i == axis else slice(None) for i in range(dim))
        slices_mid = tuple(slice(1, -1, 2) if i == axis else slice(None) for i in range(dim))
        uf[slices_mid] = (uf[slices_l] + uf[slices_r]) / 2
    return uf


def main():
    nx = 5
    L = 2 * np.pi

    x = torch.linspace(0, L, nx)
    u = torch.sin(x) + x

    uf = upscale_grid_nodes(upscale_grid_nodes(u))
    xf = torch.linspace(0, L, len(uf))

    fig, ax = plt.subplots()
    ax.plot(x.numpy(), u.numpy())
    ax.plot(xf.numpy(), uf.numpy(), 'o')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u$')
    plt.show()

if __name__ == '__main__':
    main()
