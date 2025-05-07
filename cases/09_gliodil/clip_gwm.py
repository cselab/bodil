#!/usr/bin/env python

import argparse
import numpy as np

from utils import read_vtk, dump_vtk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_vtk', help='path of input vtk file to clip')
    parser.add_argument('out_vtk', help='output path to truncated vtk')
    parser.add_argument('tumor_pos', nargs=3, type=float,
                        help='position of corner to truncate. sign sets the direction.')
    args = parser.parse_args()

    x0, y0, z0 = args.tumor_pos
    sx = np.sign(x0)
    sy = np.sign(y0)
    sz = np.sign(z0)

    out = args.out_vtk

    u, spacing, origin, varname = read_vtk(args.path_vtk)

    dx, dy, dz = spacing
    nx, ny, nz = u.shape
    Lx = nx * dx
    Ly = ny * dy
    Lz = nz * dz

    x = np.linspace(0, Lx, nx, endpoint=False)
    y = np.linspace(0, Ly, ny, endpoint=False)
    z = np.linspace(0, Lz, nz, endpoint=False)

    print(Lx, Ly, Lz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    idx = np.argwhere(np.logical_and(sx * X > x0, np.logical_and(sy * Y > y0, sz * Z > z0)))

    u[*idx.T] = 0

    dump_vtk(u, *spacing, origin, path=out, varname=varname)

if __name__ == '__main__':
    main()
