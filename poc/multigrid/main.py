#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# Code adapted from https://github.com/cselab/odil/blob/main/src/odil/core.py

def interp_to_finer(u, loc=None, depth=1):
    '''
    Interpolates a field to a finer grid.

    u: `array`
        Input field.
    loc: `str`
        Location of value in cell, one character per direction.
        'c': cell, new size `2 * n`;
        'n': node, new size `2 * (n - 1) + 1`;
        '.': none, new size `n`.
    depth: `int`
        Number of repetitions.
    '''
    if depth == 0:
        return u

    assert len(loc) == len(u.shape)
    for l in loc:
        assert l in 'cn.', "Invalid loc={}".format(loc)
    dim = len(u.shape)

    # Add padding depending on value location:
    # 'c': linear extrapolation,
    # 'n': no padding,
    # '.': no padding.
    pad_width = [(1, 1) if l == 'c' else (0, 0) for l in loc]
    ur = np.pad(u, pad_width=pad_width, mode='reflect')
    us = np.pad(u, pad_width=pad_width, mode='symmetric')
    upad = 2 * us - ur


    def term(*dd, ww=None):
        dd = [tuple(-v for v in d) for d in dd]
        return sum(w * np.roll(upad, d, range(dim))
                   for d, w in zip(dd, ww) if w) / sum(ww)

    # Offsets of nodes of a dim-dimensional cube.
    dd = np.meshgrid(*[[0] if l == '.' else [0, 1] for l in loc],
                     indexing='ij')
    dshape = tuple(1 if l == '.' else 2 for l in loc)
    # Example dim=2: dd = [(0,0), (0,1), (1,0), (1,1)].
    dd = np.reshape(dd, (dim, -1)).T
    # Indices with location in node.
    sn = [i for i, l in enumerate(loc) if l == 'n']
    # Indices with location in cell.
    sc = [i for i, l in enumerate(loc) if l == 'c']

    def weight(r, d):
        return (3**(sum(1 - abs(r - d)[sc]))  #
                if np.all((r - d)[sn] <= 0) else 0)

    uu = [term(*dd, ww=[weight(r, d) for r in dd]) for d in dd]
    res = np.stack(uu)
    res = np.reshape(res, dshape + upad.shape)
    for i in range(dim):
        res = [res[i] for i in range(res.shape[0])]
        res = np.stack(res, axis=dim + i)
    res = np.reshape(res,
                     tuple(s * d for s, d in zip(upad.shape, dshape)))
    # Remove edges.
    oslice = {'n': slice(0, -1), 'c': slice(1, -3), '.': slice(0, None)}
    res = res[tuple(oslice[l] for l in loc)]

    return interp_to_finer(res, loc, depth - 1)


def main():
    nx = 5
    L = 2 * np.pi

    x = np.linspace(0, L, nx)
    u = np.sin(x)

    uf = interp_to_finer(u, loc='n', depth=2)
    xf = np.linspace(0, L, len(uf))

    fig, ax = plt.subplots()
    ax.plot(x, u)
    ax.plot(xf, uf, 'o')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u$')
    plt.show()

if __name__ == '__main__':
    main()
