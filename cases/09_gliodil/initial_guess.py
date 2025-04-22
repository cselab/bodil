#!/usr/bin/env python

import argparse
import numpy as np
import os
import torch
from scipy.ndimage import zoom

from prepare_data import load_data, SEG_CODE

def center_of_mass(X, Y, Z, mask):
    V = np.sum(mask)
    x = np.sum(X * mask) / V
    y = np.sum(Y * mask) / V
    z = np.sum(Z * mask) / V
    return x, y, z

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

def dice_score(maska, maskb):
    A = np.sum(maska)
    B = np.sum(maskb)
    AandB = np.sum(np.logical_and(maska, maskb))
    return 2 * AandB / (A + B)

def compute_dice_scores(u, seg, th_lo, th_hi):
    segu = np.where(u < th_lo, SEG_CODE.healthy, np.where(u < th_hi, SEG_CODE.edema, SEG_CODE.core))

    mask_core_data = np.where(seg == SEG_CODE.core, 1.0, 0.0)
    mask_core_sim = np.where(segu == SEG_CODE.core, 1.0, 0.0)

    mask_edema_data = np.where(seg == SEG_CODE.edema, 1.0, 0.0)
    mask_edema_sim = np.where(segu == SEG_CODE.edema, 1.0, 0.0)

    return dice_score(mask_core_data, mask_core_sim), dice_score(mask_edema_data, mask_edema_sim)


def get_initial_guess(data_path, Nx, Ny, Nz, trim_scale, Nt_ODIL, verbose):
    """
    Arguments:
        data_path: path to directory that contains the nii GM, WM and SEG files
        Nx, Ny, Nz: grid resolution
        trim_scale: scale to select the trimmed region (1 = smallest rectangle, should be > 1
        Nt_ODIL: Number of snapshots to get as initial guess for ODIL
        verbose: if True, will print diagnostics on stdout

    Return:
        tend: simulation time that maximizes the dice score
        uodil: interpolated values of the field
        params: dictionary of parameters that were used in simulation
    """
    meta_data, raw_data, trimmed_data = load_data(data_path, trim_scale)
    # adjust data
    trimmed_shape = trimmed_data['seg'].shape
    seg = zoom(trimmed_data['seg'], (Nx/trimmed_shape[0], Ny/trimmed_shape[1], Nz/trimmed_shape[2]), order=0).clip(0.0)
    gm  = zoom(trimmed_data['gm'],  (Nx/trimmed_shape[0], Ny/trimmed_shape[1], Nz/trimmed_shape[2]), order=0).clip(0.0)
    wm  = zoom(trimmed_data['wm'],  (Nx/trimmed_shape[0], Ny/trimmed_shape[1], Nz/trimmed_shape[2]), order=0).clip(0.0)
    assert gm.shape[0] == Nx and  gm.shape[1] == Ny and  gm.shape[2] == Nz
    assert len(np.unique(seg)) <= 3

    tend = 100
    maxsteps = 1000
    dt = tend / maxsteps

    Lx, Ly, Lz = trimmed_shape # mm
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    z = np.linspace(0, Lz, Nz, endpoint=False)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    x0, y0, z0 = center_of_mass(X, Y, Z, np.where(seg == SEG_CODE.core, 1.0, 0.0))

    volume_edema = np.sum(np.where(seg == SEG_CODE.edema, 1.0, 0.0))
    volume_core  = np.sum(np.where(seg == SEG_CODE.core , 1.0, 0.0))

    wDcore = volume_core / (volume_core + volume_edema)
    wDedema = volume_edema / (volume_core + volume_edema)

    # see GliODIL paper
    Dw = volume_edema / volume_core
    rho = 0.12
    Dg = Dw / 100
    th_lo = 0.3
    th_hi = 0.65

    dtmax = max([dx, dy, dz])**2 / (2 * Dw)
    assert dt < dtmax

    D = compute_D(gm=gm, wm=wm, Dg=Dg, Dw=Dw)

    u = initial_density(X, Y, Z, x0, y0, z0)

    trace_u = []
    trace_scores = []

    endstep = maxsteps
    for step in range(maxsteps):
        Dcore, Dedema = compute_dice_scores(u, seg, th_lo=th_lo, th_hi=th_hi)
        score = wDcore * Dcore + wDedema * Dedema

        trace_u.append(u.copy())
        trace_scores.append(score)

        #if len(trace_scores) > 10 and trace_scores[-3] > trace_scores[-2] and trace_scores[-3] > trace_scores[-1] and score > 0:
        #    endstep = step
        #    break

        u = advance(u, D, rho, dx, dy, dz, dt)

    endstep = np.argmax(trace_scores)

    tend = endstep * dt

    if verbose:
        print(f"Forward problem: maximized dice score at t={tend} ({endstep} steps)")

    # linear interpolation in time
    trace_u = np.array(trace_u)
    tsim  = np.linspace(0, tend, endstep+1, endpoint=True)
    todil = np.linspace(0, tend, Nt_ODIL, endpoint=True)

    dtsim = tsim[1] - tsim[0]
    assert dt == dtsim

    uodil = np.empty((Nt_ODIL, Nx, Ny, Nz))

    for j, t in enumerate(todil):
        i0 = int(t / dtsim)
        i1 = min([i0 + 1, endstep])
        tlo = tsim[i0]
        thi = tsim[i1]
        assert tlo <= t and t <= thi
        w0 = (t - tlo) / dtsim
        w1 = 1 - w0
        uodil[j] = w0 * trace_u[i0] + w1 * trace_u[i1]

    params = {
        'Dw': Dw,
        'Dg': Dg,
        'rho': rho,
        'dx': dx,
        'dy': dy,
        'dz': dz,
        'x0': x0,
        'y0': y0,
        'z0': z0,
        'th_lo': th_lo,
        'th_hi': th_hi
    }

    return tend, uodil, params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to .nii files')
    parser.add_argument('--NxNyNz', type=int, nargs=3, default=[64, 64, 64], help='grid size (Nx, Ny, Nz)')
    parser.add_argument('--out-dir', type=str, default='out_init_guess', help='output directory')
    args = parser.parse_args()

    Nx, Ny, Nz = args.NxNyNz
    Nt_ODIL = 33

    tend, uodil, params = get_initial_guess(args.data_path, Nx, Ny, Nz, trim_scale=1.5, Nt_ODIL=Nt_ODIL, verbose=True)

if __name__ == '__main__':
    main()
