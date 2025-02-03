#!/usr/bin/env python3

import numpy as np


def rotational(x, y, z, mod, *, k):
    return k * y, -k * x, 0


def vortex(x, y, z, mod, *, k, sigma, x0=0.0, y0=0.0, z0=0.0, norm=0):
    dx = x - x0
    dy = y - y0
    dz = z - z0
    r2 = dx**2 + dy**2 + dz**2
    if norm:
        r = (r2 + 1e-8)**0.5
        dx /= r
        dy /= r
        dz /= r
    factor = k * mod.exp(-r2 / (2 * sigma**2))
    return rotational(dx, dy, dz, mod=mod, k=factor)


def get_vorticity(func, x, y, z, mod, h=1e-8):
    '''
    Computes vorticity field from flow using finite differences.

    func: `callable`
        Flow function from `get_flow()`.
    x,y,z: `np.ndarray`
        Target points.
    h: `float`
        Step size.
    '''
    _, vxm, wxm = func(x - h, y, z, mod=mod)
    _, vxp, wxp = func(x + h, y, z, mod=mod)
    uym, _, wym = func(x, y - h, z, mod=mod)
    uyp, _, wyp = func(x, y + h, z, mod=mod)
    uzm, vzm, _ = func(x, y, z - h, mod=mod)
    uzp, vzp, _ = func(x, y, z + h, mod=mod)
    v_x = (vxp - vxm) / (2 * h)
    w_x = (wxp - wxm) / (2 * h)
    u_y = (uyp - uym) / (2 * h)
    w_y = (wyp - wym) / (2 * h)
    u_z = (uzp - uzm) / (2 * h)
    v_z = (vzp - vzm) / (2 * h)
    omegax = w_y - v_z
    omegay = u_z - w_x
    omegaz = v_x - u_y
    return omegax, omegay, omegaz


def make_periodic(x, L, mod):
    x -= L * mod.floor(x / L)
    return x


def interpolate(field, h, x, y, z, mod):
    nz, ny, nx = field.shape
    dtype = x.dtype
    Lx = nx * h[0]
    Ly = ny * h[1]
    Lz = nz * h[2]
    #x = make_periodic(x, Lx, mod)
    #y = make_periodic(y, Ly, mod)
    #z = make_periodic(z, Lz, mod)
    x = mod.clip(x, 0, Lx)
    y = mod.clip(y, 0, Ly)
    z = mod.clip(z, 0, Lz)

    ix = mod.cast(mod.stop_gradient(x / h[0]), mod.int32)
    iy = mod.cast(mod.stop_gradient(y / h[1]), mod.int32)
    iz = mod.cast(mod.stop_gradient(z / h[2]), mod.int32)
    ix = mod.maximum(0, mod.minimum(nx - 1, ix))
    iy = mod.maximum(0, mod.minimum(ny - 1, iy))
    iz = mod.maximum(0, mod.minimum(nz - 1, iz))
    ixp = mod.minimum(nx - 1, ix + 1)
    iyp = mod.minimum(ny - 1, iy + 1)
    izp = mod.minimum(nz - 1, iz + 1)

    f000 = mod.gather_nd(field, mod.stack((iz, iy, ix), axis=-1))
    f001 = mod.gather_nd(field, mod.stack((iz, iy, ixp), axis=-1))
    f010 = mod.gather_nd(field, mod.stack((iz, iyp, ix), axis=-1))
    f011 = mod.gather_nd(field, mod.stack((iz, iyp, ixp), axis=-1))
    f100 = mod.gather_nd(field, mod.stack((izp, iy, ix), axis=-1))
    f101 = mod.gather_nd(field, mod.stack((izp, iy, ixp), axis=-1))
    f110 = mod.gather_nd(field, mod.stack((izp, iyp, ix), axis=-1))
    f111 = mod.gather_nd(field, mod.stack((izp, iyp, ixp), axis=-1))

    lx = (x - mod.cast(ix, dtype) * h[0]) / h[0]
    ly = (y - mod.cast(iy, dtype) * h[1]) / h[1]
    lz = (z - mod.cast(iz, dtype) * h[2]) / h[2]

    f00 = (1 - lx) * f000 + lx * f001
    f01 = (1 - lx) * f010 + lx * f011
    f10 = (1 - lx) * f100 + lx * f101
    f11 = (1 - lx) * f110 + lx * f111

    f0 = (1 - ly) * f00 + ly * f01
    f1 = (1 - ly) * f10 + ly * f11

    return (1 - lz) * f0 + lz * f1


class CubeTop1:

    def __init__(self, path, mod, dtype):
        v0 = np.load(path)
        dim, nz, ny, nx = v0.shape
        assert nx == ny
        assert nx == nz
        self.L = 1.
        self.h = 3 * [1. / nx]
        self.v0 = mod.array(v0, dtype=dtype)

    def get_velocity(self, x, y, z, k, mod):
        assert len(k) == self.num_vortices()
        L = self.L
        h = self.h
        v0 = self.v0

        def comp(c):
            return \
                k[0] * interpolate(v0[c,      ...], h, x, y,     z, mod) + \
                k[1] * interpolate(v0[c,      ...], h, x, y, L - z, mod) + \
                k[2] * interpolate(v0[(c+2)%3,...], h, y, z,     x, mod) + \
                k[3] * interpolate(v0[(c+2)%3,...], h, y, z, L - x, mod) + \
                k[4] * interpolate(v0[(c+1)%3,...], h, z, x,     y, mod) + \
                k[5] * interpolate(v0[(c+1)%3,...], h, z, x, L - y, mod)

        return comp(0), comp(1), comp(2)

    def num_vortices(self):
        return 6


class CubeNoLid:

    def __init__(self, path_x0, path_z0, mod, dtype):
        vx0 = np.load(path_x0)
        vz0 = np.load(path_z0)
        dim, nz, ny, nx = vx0.shape
        assert vx0.shape == vz0.shape
        assert nx == ny and nx == nz
        self.L = 1.
        self.h = 3 * [1. / nx]
        self.vx0 = mod.array(vx0, dtype=dtype)
        self.vz0 = mod.array(vz0, dtype=dtype)

    def get_velocity(self, x, y, z, k, mod):
        assert len(k) == self.num_vortices()
        L = self.L
        h = self.h
        vx0 = self.vx0
        vz0 = self.vz0

        def comp(c, cy):
            return \
                k[0] * interpolate(vx0[c],  h,     x, y, z, mod) + \
                k[1] * interpolate(vx0[c],  h, L - x, y, z, mod) + \
                k[2] * interpolate(vx0[cy], h,     y, x, z, mod) + \
                k[3] * interpolate(vx0[cy], h, L - y, x, z, mod) + \
                k[4] * interpolate(vz0[c],  h,     x, y, z, mod)

        return comp(0, 1), comp(1, 0), comp(2, 2)

    def num_vortices(self):
        return 5
