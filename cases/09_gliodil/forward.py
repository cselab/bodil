#!/usr/bin/env python

import argparse
import nibabel as nib
import numpy as np
import os
from scipy.ndimage import zoom

from prepare_data import load_data, SEG_CODE, get_grid_spacing
from utils import dump_vtk

def get_matter_portions(gm, wm, threshold):
    """
    threshold: crop density to zero when wm + gm <= threshold
    """
    def get_tilda(a, b, axis):
        val = np.where(np.logical_and(np.roll(a + b, -1, axis=axis) >= threshold, a + b >= threshold),
                       (np.roll(a, -1, axis=axis) + a) / 2,
                       0.0)
        return val

    return {
        'wm_t_x': get_tilda(wm, gm, 0),
        'wm_t_y': get_tilda(wm, gm, 1),
        'wm_t_z': get_tilda(wm, gm, 2),
        'gm_t_x': get_tilda(gm, wm, 0),
        'gm_t_y': get_tilda(gm, wm, 1),
        'gm_t_z': get_tilda(gm, wm, 2)
    }

def compute_D(gm, wm, Dg, Dw):
    matter = get_matter_portions(gm=gm, wm=wm, threshold=0.1)

    Dxm = Dw * matter['wm_t_x'] + Dg * matter['gm_t_x']
    Dym = Dw * matter['wm_t_y'] + Dg * matter['gm_t_y']
    Dzm = Dw * matter['wm_t_z'] + Dg * matter['gm_t_z']

    Dxp = np.roll(Dxm, +1, 0)
    Dyp = np.roll(Dym, +1, 1)
    Dzp = np.roll(Dzm, +1, 2)

    return {'xm': Dxm, 'ym': Dym, 'zm': Dzm,
            'xp': Dxp, 'yp': Dyp, 'zp': Dzp}

def advance(u, D, rho, dx, dy, dz, dt):
    uxx = (D["xp"] * (np.roll(u,1,axis=0) - u) - D["xm"] * (u - np.roll(u,-1,axis=0))) / dx**2
    uyy = (D["yp"] * (np.roll(u,1,axis=1) - u) - D["ym"] * (u - np.roll(u,-1,axis=1))) / dy**2
    uzz = (D["zp"] * (np.roll(u,1,axis=2) - u) - D["zm"] * (u - np.roll(u,-1,axis=2))) / dz**2
    du = (uxx + uyy + uzz + rho * u * (1 - u)) * dt
    return u + du

def initial_density(X, Y, Z, x0, y0, z0, mod=np):
    dsq = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2

    # following original gliodil code
    M = 1500.0
    Dt = 15.0

    u0 = M / (4 * np.pi * Dt)**(3/2) * mod.exp(-dsq / (4 * Dt))
    u0 = mod.where(u0 > 0.1, mod.where(u0 < 1.0, u0, 1.0), 0.0)
    return u0

def run_forward(data_path, Nt, Nx, Ny, Nz, out_dir,
                xyz0, rho, Dg, Dw,
                trim_scale=1.5,
                verbose=True, tend=50.0,
                matter_th=0.1, dump_raw_to_vtk=False,
                dump_results_mode='last_only'):
    """
    Arguments:
        data_path: path to patient data
        Nt, Nx, Ny, Nz: grid size
        out_dir: output directory
        trim_scale: scale to choose the domain. 1.5 means 50% larger than the ,inimal one. must be larger than 1.
        xyz0: initial position of the tumor.
        rho: growth rate
        Dg, Dw: diffusion constant in gray and white matter, respectively
        verbose: if True, reports additional information to stdout
        tend: simulation time scale
        matter_th: threshold to condiser what is matter or not.
        dump_raw_to_vtk: if true, dump the raw data to vtk at the start of the simulation. Useful to compare with ODIL solution.
        dump_results_mode: what to dump at the end of the training loop:
            * none: do not dump anything
            * all: dump all time steps
            * last_only: dump only the last time step
    """

    os.makedirs(out_dir, exist_ok=True)

    meta_data, raw_data, trimmed_data = load_data(data_path, trim_scale)
    dx_raw, dy_raw, dz_raw = get_grid_spacing(meta_data['nifti_header'])

    if dump_raw_to_vtk:
        dump_vtk(raw_data['gm'], dx=dx_raw, dy=dy_raw, dz=dz_raw, origin=(0.0, 0.0, 0.0), varname='gm',
                 path=os.path.join(out_dir, 'gm.vtk'))
        dump_vtk(raw_data['wm'], dx=dx_raw, dy=dy_raw, dz=dz_raw, origin=(0.0, 0.0, 0.0), varname='wm',
                 path=os.path.join(out_dir, 'wm.vtk'))
        dump_vtk(raw_data['seg'], dx=dx_raw, dy=dy_raw, dz=dz_raw, origin=(0.0, 0.0, 0.0), varname='seg',
                 path=os.path.join(out_dir, 'seg.vtk'))

    # adjust data
    trimmed_shape = trimmed_data['seg'].shape
    seg = zoom(trimmed_data['seg'], (Nx/trimmed_shape[0], Ny/trimmed_shape[1], Nz/trimmed_shape[2]), order=0).clip(0.0)
    gm  = zoom(trimmed_data['gm'],  (Nx/trimmed_shape[0], Ny/trimmed_shape[1], Nz/trimmed_shape[2]), order=0).clip(0.0)
    wm  = zoom(trimmed_data['wm'],  (Nx/trimmed_shape[0], Ny/trimmed_shape[1], Nz/trimmed_shape[2]), order=0).clip(0.0)
    assert gm.shape[0] == Nx and  gm.shape[1] == Ny and  gm.shape[2] == Nz
    assert len(np.unique(seg)) <= 3

    Nx_raw, Ny_raw, Nz_raw = trimmed_shape
    Lx = Nx_raw * dx_raw
    Ly = Ny_raw * dy_raw
    Lz = Nz_raw * dz_raw

    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    dt = tend / Nt

    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    z = np.linspace(0, Lz, Nz, endpoint=False)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    assert x[1] - x[0] == dx

    if verbose:
        print(f"dx = {dx:.2f}mm, dy = {dy:.2f}mm, dz = {dz:.2f}mm")

    matter = get_matter_portions(gm, wm, threshold=matter_th)

    x0, y0, z0 = xyz0

    D = compute_D(gm=gm, wm=wm, Dg=Dg, Dw=Dw)
    u = initial_density(X, Y, Z, x0, y0, z0)
    trace_u = [u.copy()]

    for step in range(Nt):
        u = advance(u, D, rho, dx, dy, dz, dt)
        trace_u.append(u.copy())


    if dump_results_mode == 'all':
        for it in range(Nt):
            dump_vtk(trace_u[it], dx, dy, dz, origin=meta_data['crop_offset'],
                     path=os.path.join(out_dir, f'seg_{it:04d}.vtk'))
    elif dump_results_mode == 'last_only':
        dump_vtk(trace_u[-1], dx, dy, dz, origin=meta_data['crop_offset'],
                 path=os.path.join(out_dir, f'seg_final.vtk'))
    elif dump_results_mode == 'none':
        pass
    else:
        raise ValueError(f'unknown dump_results_mode flag, got {dump_results_mode}')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to directory containing .nii files')
    parser.add_argument('--NtNxNyNz', type=int, nargs=4, default=[257, 64, 64, 64], help='odil grid size (Nt, Nx, Ny, Nz)')
    parser.add_argument('--out-dir', type=str, default='out_forward', help='output directory')
    parser.add_argument('--xyz0', type=float, nargs=3, default=[55.0, 70.0, 55.0], help='Initial tumor position')
    parser.add_argument('--Dg', type=float, default=0.001, help='diffusion coefficient in grey matter')
    parser.add_argument('--Dw', type=float, default=0.1, help='diffusion coefficient in white matter')
    parser.add_argument('--rho', type=float, default=0.12, help='reaction rate')
    args = parser.parse_args()

    Nt, Nx, Ny, Nz = args.NtNxNyNz
    out_dir = args.out_dir

    rho = args.rho
    Dg = args.Dg
    Dw = args.Dw

    run_forward(data_path=args.data_path,
                Nt=Nt, Nx=Nx, Ny=Ny, Nz=Nz,
                rho=rho, Dg=Dg, Dw=Dw, out_dir=out_dir,
                dump_raw_to_vtk=True,
                xyz0=args.xyz0,
                dump_results_mode='all')

if __name__ == '__main__':
    main()
