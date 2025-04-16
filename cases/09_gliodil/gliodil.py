#!/usr/bin/env python

import argparse
import numpy as np
import os
import torch
from scipy.ndimage import zoom

from prepare_data import load_data, SEG_CODE
from uq_odil.multigrid import MultigridField

def get_matter_portions(gm, wm, threshold, device):
    """
    threshold: crop density to zero when wm + gm <= threshold
    """
    def get_tilda(a, b, axis):
        val = np.where(np.logical_and(np.roll(a + b, -1, axis=axis) >= threshold, a + b >= threshold),
                       (np.roll(a, -1, axis=axis) + a) / 2,
                       0.0)
        val = torch.from_numpy(val)
        val.to(device)
        return val

    return {
        'wm_t_x': get_tilda(wm, gm, 0),
        'wm_t_y': get_tilda(wm, gm, 1),
        'wm_t_z': get_tilda(wm, gm, 2),
        'gm_t_x': get_tilda(gm, wm, 0),
        'gm_t_y': get_tilda(gm, wm, 1),
        'gm_t_z': get_tilda(gm, wm, 2)
    }


def run_gliodil(data_path, Nt, Nx, Ny, Nz, device,
                trim_scale=1.5,
                num_epochs=1000, lr=1e-2, report_every=10,
                verbose=True, tend=64.0, lambda_pde=1, lambda_ic=1):

    meta_data, raw_data, trimmed_data = load_data(data_path, trim_scale)

    # adjust data
    trimmed_shape = trimmed_data['seg'].shape
    seg = zoom(trimmed_data['seg'], (Nx/trimmed_shape[0], Ny/trimmed_shape[1], Nz/trimmed_shape[2]), order=0).clip(0.0)
    gm  = zoom(trimmed_data['gm'],  (Nx/trimmed_shape[0], Ny/trimmed_shape[1], Nz/trimmed_shape[2]), order=0).clip(0.0)
    wm  = zoom(trimmed_data['wm'],  (Nx/trimmed_shape[0], Ny/trimmed_shape[1], Nz/trimmed_shape[2]), order=0).clip(0.0)
    assert gm.shape[0] == Nx and  gm.shape[1] == Ny and  gm.shape[2] == Nz
    assert len(np.unique(seg)) <= 3

    Lx, Ly, Lz = trimmed_shape # mm
    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz
    dt = tend / Nt

    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    z = np.linspace(0, Lz, Nz, endpoint=False)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    X_ = torch.from_numpy(X).to(device)
    Y_ = torch.from_numpy(Y).to(device)
    Z_ = torch.from_numpy(Z).to(device)

    assert x[1] - x[0] == dx

    print(f"dx = {dx}mm, dy = {dy}mm, dz = {dz}mm")

    matter = get_matter_portions(gm, wm, threshold=0.1, device=device)
    seg_ = torch.from_numpy(seg).to(device)

    mask_core = torch.where(seg_ == SEG_CODE.core, 1.0, 0.0)
    mask_edema = torch.where(seg_ == SEG_CODE.edema, 1.0, 0.0)

    def compute_pde_loss(u, Dw, Dg, rho):
        # [c]urrent time
        uc = u[:-1,:,:,:]
        uc_xm = torch.roll(uc, +1, 0)
        uc_xp = torch.roll(uc, -1, 0)
        uc_ym = torch.roll(uc, +1, 1)
        uc_yp = torch.roll(uc, -1, 1)
        uc_zm = torch.roll(uc, +1, 2)
        uc_zp = torch.roll(uc, -1, 2)

        # [n]ext time
        un = u[1:,:,:,:]
        un_xm = torch.roll(un, +1, 0)
        un_xp = torch.roll(un, -1, 0)
        un_ym = torch.roll(un, +1, 1)
        un_yp = torch.roll(un, -1, 1)
        un_zm = torch.roll(un, +1, 2)
        un_zp = torch.roll(un, -1, 2)

        # diffusion coefficient
        Dxm = Dw * matter['wm_t_x'] + Dg * matter['gm_t_x']
        Dym = Dw * matter['wm_t_y'] + Dg * matter['gm_t_y']
        Dzm = Dw * matter['wm_t_z'] + Dg * matter['gm_t_z']

        Dxp = torch.roll(Dxm, +1, 0)
        Dyp = torch.roll(Dym, +1, 1)
        Dzp = torch.roll(Dzm, +1, 2)

        # diffusion term
        uc_xx = (Dxp * (uc_xp - uc) - Dxm * (uc - uc_xm)) / dx**2
        uc_yy = (Dyp * (uc_yp - uc) - Dym * (uc - uc_ym)) / dy**2
        uc_zz = (Dzp * (uc_zp - uc) - Dzm * (uc - uc_zm)) / dz**2

        un_xx = (Dxp * (un_xp - un) - Dxm * (un - un_xm)) / dx**2
        un_yy = (Dyp * (un_yp - un) - Dym * (un - un_ym)) / dy**2
        un_zz = (Dzp * (un_zp - un) - Dzm * (un - un_zm)) / dz**2

        diff_term = (uc_xx + un_xx +
                     uc_yy + un_yy +
                     uc_zz + un_zz) / 2.0

        # reaction term
        reac_term = rho / 2.0 * (torch.abs(uc) * (1 - uc) +
                                 torch.abs(un) * (1 - un))

        # PDE loss
        u_t = (un - uc) / dt

        PDE_res = u_t - diff_term - reac_term
        return lambda_pde * torch.mean(PDE_res**2)

    def compute_ic_loss(u, x0, y0, z0):
        dsq = (X_ - x0)**2 + (Y_ - y0)**2 + (Z_ - z0)**2

        # following original gliodil code
        M = 1500.0
        Dt = 15.0

        u0 = M / (4 * np.pi * Dt)**(3/2) * torch.exp(-dsq / (4 * Dt))
        u0 = torch.where(u0 > 0.1, torch.where(u0 < 1.0, u0, 1.0), 0.0)
        res_ic = u - u0

        return lambda_ic * torch.mean(res_ic**2)

    def compute_data_loss(u, th_core, th_edema_lo, th_edema_hi):
        sigma_data = 1/50
        eps = 1e-6
        uend = u[-1,:,:,:]

        alpha_core = torch.sigmoid((uend - th_core) / sigma_data)
        alpha_core = torch.minimum(torch.full_like(alpha_core, 1-eps), alpha_core)
        alpha_core = torch.maximum(torch.full_like(alpha_core,   eps), alpha_core)

        alpha_edema = torch.sigmoid((uend - th_edema_lo) / sigma_data) - (1 - torch.sigmoid((th_edema_hi - uend) / sigma_data))
        alpha_edema = torch.minimum(torch.full_like(alpha_edema, 1-eps), alpha_edema)
        alpha_edema = torch.maximum(torch.full_like(alpha_edema,   eps), alpha_edema)

        neg_loss  = mask_core  * torch.log(alpha_core)  + (1-mask_core)  * torch.log(1-alpha_core)
        neg_loss += mask_edema * torch.log(alpha_edema) + (1-mask_edema) * torch.log(1-alpha_edema)

        return -torch.mean(neg_loss)


    # initial guess
    u0 = torch.zeros((Nt, Nx, Ny, Nz)) + 0.5
    depth = int(np.log(min([Nt, Nx, Ny, Nz])) / np.log(2))
    if verbose:
        print(f"Multigrid depth = {depth}")
    mg = MultigridField(u0, loc='nppp', depth=depth)
    mg.to(device)
    mg.set_requires_grad()

    # TODO
    Dw = 0.05
    Dg = 0.01
    rho = 0.01
    x0 = Lx/2
    y0 = Ly/2
    z0 = Lz/2
    th_core = 0.7
    th_edema_lo = 0.3
    th_edema_hi = 0.7

    optim = torch.optim.Adam(mg.params(), lr=lr)

    epochs = list(range(num_epochs))
    pde_losses = []
    data_losses = []
    ic_losses = []
    losses = []

    for epoch in epochs:
        optim.zero_grad()
        u = mg.get()
        ploss = compute_pde_loss(u, Dw=Dw, Dg=Dg, rho=rho)
        dloss = compute_data_loss(u, th_core, th_edema_lo, th_edema_hi)
        iloss = compute_ic_loss(u, x0, y0, z0)
        loss = ploss + iloss + dloss
        loss.backward()
        optim.step()

        l = loss.item()
        pde_losses.append(ploss.item())
        data_losses.append(dloss.item())
        ic_losses.append(iloss.item())
        losses.append(l)

        #scheduler.step(l)

        if verbose and epoch % report_every == 0:
            print(f"epoch {epoch:06d} loss {l:.4e}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to .nii files')
    parser.add_argument('--NtNxNyNz', type=int, nargs=4, default=[33, 64, 64, 64], help='odil grid size (Nt, Nx, Ny, Nz)')
    args = parser.parse_args()

    Nt, Nx, Ny, Nz = args.NtNxNyNz

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_gliodil(data_path=args.data_path,
                Nt=Nt, Nx=Nx, Ny=Ny, Nz=Nz,
                device=device)

if __name__ == '__main__':
    main()
