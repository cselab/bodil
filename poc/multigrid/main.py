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



def strided_convolution_nd(input_array, kernel, stride):
    """
    Perform strided convolution in n dimensions with 'VALID' padding.

    Parameters:
    input_array (np.ndarray): The input array of shape (D1, D2, ..., Dn)
    kernel (np.ndarray): The kernel (filter) of shape (K1, K2, ..., Kn)
    stride (tuple): The stride for each dimension (S1, S2, ..., Sn)

    Returns:
    np.ndarray: The result of the convolution.
    """
    # Ensure input validity
    if len(input_array.shape) != len(kernel.shape) or len(stride) != len(input_array.shape):
        raise ValueError("Input, kernel, and stride must have the same number of dimensions")

    # Compute output shape for 'VALID' padding
    output_shape = tuple(
        (input_array.shape[i] - kernel.shape[i]) // stride[i] + 1 for i in range(len(input_array.shape))
    )

    # Initialize output array
    output = np.zeros(output_shape, dtype=input_array.dtype)

    # Iterate over output indices
    for idx in product(*[range(s) for s in output_shape]):
        # Compute the starting index for each dimension
        start_idx = tuple(idx[i] * stride[i] for i in range(len(idx)))

        # Extract sub-array and compute convolution
        sub_array = input_array[
            tuple(slice(start_idx[i], start_idx[i] + kernel.shape[i]) for i in range(len(start_idx)))
        ]
        output[idx] = np.sum(sub_array * kernel)

    return output

def restrict_to_coarser(u, loc=None, depth=1):
    '''
    Restricts a field to a coarser grid.

    u: `array`
        Input field.
    loc: `str`
        Location of value in cell, one character per direction.
        'c': cell, new size `n // 2`;
        'n': node, new size `(n - 1) // 2 + 1`;
        '.': none, new size `n`.
    method: `str`
        Restriction method.
        'conv': using the convolution
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
    # 'c': no padding,
    # 'n': linear extrapolation, combined with the [1,2,1] kernel
    #      implements the identity condition on the boundaries
    # '.': no padding.
    pad_width = [(1, 1) if l == 'n' else (0, 0) for l in loc]
    ur = np.pad(u, pad_width=pad_width, mode='reflect')
    us = np.pad(u, pad_width=pad_width, mode='symmetric')
    upad = 2 * us - ur


    # Convolution weights.
    wnode = np.array([1, 2, 1]) * 0.25
    wcell = np.array([1, 1]) * 0.5
    wnone = np.array([1.])
    wloc = {'n': wnode, 'c': wcell, '.': wnone}
    w = wloc[loc[0]]
    for i in range(1, dim):
        w = np.kron(wloc[loc[i]], w[..., None])
    w = w.astype(u.dtype)
    res = strided_convolution_nd(upad, kernel=w, stride=tuple([2] * dim))

    return restrict_to_coarser(res, loc, depth - 1)



def main():
    nx = 8
    L = 2 * np.pi

    x = np.linspace(0, L, nx)
    u = np.sin(x)

    uf = interp_to_finer(u, loc='n', depth=2)
    xf = np.linspace(0, L, len(uf))

    uc = restrict_to_coarser(uf, loc='n', depth=2)
    xc = np.linspace(0, L, len(uc))

    fig, ax = plt.subplots()
    ax.plot(x, u)
    ax.plot(xf, uf, 'o')
    ax.plot(xc, uc, 'o')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u$')
    plt.show()

if __name__ == '__main__':
    main()
