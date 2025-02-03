#!/usr/bin/env python3

import numpy as np
import torch

def interpolate(field, h, x, y, z):
    nz, ny, nx = field.shape
    dtype = x.dtype
    Lx = nx * h[0]
    Ly = ny * h[1]
    Lz = nz * h[2]
    x = torch.clip(x, 0, Lx)
    y = torch.clip(y, 0, Ly)
    z = torch.clip(z, 0, Lz)

    with torch.no_grad():
        ix = (x / h[0]).to(torch.int32)
        iy = (y / h[1]).to(torch.int32)
        iz = (z / h[2]).to(torch.int32)
        ix = torch.maximum(torch.zeros_like(ix), torch.minimum(torch.full_like(ix, nx - 1), ix))
        iy = torch.maximum(torch.zeros_like(iy), torch.minimum(torch.full_like(iy, ny - 1), iy))
        iz = torch.maximum(torch.zeros_like(iz), torch.minimum(torch.full_like(iz, nz - 1), iz))
        ixp = torch.minimum(torch.full_like(ix, nx - 1), ix + 1)
        iyp = torch.minimum(torch.full_like(iy, ny - 1), iy + 1)
        izp = torch.minimum(torch.full_like(iz, nz - 1), iz + 1)

    f000 = field[iz, iy, ix]
    f001 = field[iz, iy, ixp]
    f010 = field[iz, iyp, ix]
    f011 = field[iz, iyp, ixp]
    f100 = field[izp, iy, ix]
    f101 = field[izp, iy, ixp]
    f110 = field[izp, iyp, ix]
    f111 = field[izp, iyp, ixp]

    lx = (x - ix.to(dtype) * h[0]) / h[0]
    ly = (y - iy.to(dtype) * h[1]) / h[1]
    lz = (z - iz.to(dtype) * h[2]) / h[2]

    f00 = (1 - lx) * f000 + lx * f001
    f01 = (1 - lx) * f010 + lx * f011
    f10 = (1 - lx) * f100 + lx * f101
    f11 = (1 - lx) * f110 + lx * f111

    f0 = (1 - ly) * f00 + ly * f01
    f1 = (1 - ly) * f10 + ly * f11

    return (1 - lz) * f0 + lz * f1


class CubeNoLid:

    def __init__(self, path_x0, path_z0):
        vx0 = np.load(path_x0)
        vz0 = np.load(path_z0)
        dim, nz, ny, nx = vx0.shape
        assert vx0.shape == vz0.shape
        assert nx == ny and nx == nz
        self.L = 1.
        self.h = 3 * [1. / nx]
        self.vx0 = torch.from_numpy(vx0).float()
        self.vz0 = torch.from_numpy(vz0).float()

    def get_velocity(self, x, y, z, k):
        assert len(k) == self.num_vortices()
        L = self.L
        h = self.h
        vx0 = self.vx0
        vz0 = self.vz0

        def comp(c, cy):
            return \
                k[0] * interpolate(vx0[c],  h,     x, y, z) + \
                k[1] * interpolate(vx0[c],  h, L - x, y, z) + \
                k[2] * interpolate(vx0[cy], h,     y, x, z) + \
                k[3] * interpolate(vx0[cy], h, L - y, x, z) + \
                k[4] * interpolate(vz0[c],  h,     x, y, z)

        return comp(0, 1), comp(1, 0), comp(2, 2)

    def num_vortices(self):
        return 5
