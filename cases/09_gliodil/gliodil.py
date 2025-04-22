#!/usr/bin/env python

import argparse
import nibabel as nib
import numpy as np
import os
import pandas as pd
import torch
from scipy.ndimage import zoom

from prepare_data import load_data, SEG_CODE, restore_cropped_data
from uq_odil.multigrid import MultigridField
from initial_guess import get_initial_guess

def get_matter_portions(gm, wm, threshold, device):
    """
    threshold: crop density to zero when wm + gm <= threshold
    """
    def get_tilda(a, b, axis):
        val = np.where(np.logical_and(np.roll(a + b, -1, axis=axis) >= threshold, a + b >= threshold),
                       (np.roll(a, -1, axis=axis) + a) / 2,
                       0.0)
        val = torch.from_numpy(val)
        return val.to(device)

    return {
        'wm_t_x': get_tilda(wm, gm, 0),
        'wm_t_y': get_tilda(wm, gm, 1),
        'wm_t_z': get_tilda(wm, gm, 2),
        'gm_t_x': get_tilda(gm, wm, 0),
        'gm_t_y': get_tilda(gm, wm, 1),
        'gm_t_z': get_tilda(gm, wm, 2)
    }

def dump_nii(u, th_lo, th_hi, raw_data, meta_data, trim_scale, trimmed_shape, path):
    nx, ny, nz = u.shape
    seg_lowres = np.where(u < th_lo, SEG_CODE.healthy, np.where(u < th_hi, SEG_CODE.edema, SEG_CODE.core))
    Nx, Ny, Nz = trimmed_shape

    trimmed_seg = zoom(seg_lowres, (Nx/nx, Ny/ny, Nz/nz), order=0)
    seg = restore_cropped_data(raw_data['seg'], trim_scale, trimmed_seg)

    nifti_file = nib.Nifti1Image(seg, meta_data['nifti_affine'], header=meta_data['nifti_header'])
    nib.save(nifti_file, path)

def dump_vtk(u, dx, dy, dz, path):
    nx, ny, nz = u.shape
    num_points = nx * ny * nz
    spacing = (dx, dy, dz)
    origin = (0.0, 0.0, 0.0)

    with open(path, "wb") as f:
        header = f"""# vtk DataFile Version 3.0
Binary uniform grid
BINARY
DATASET STRUCTURED_POINTS
DIMENSIONS {nx} {ny} {nz}
ORIGIN {origin[0]} {origin[1]} {origin[2]}
SPACING {spacing[0]} {spacing[1]} {spacing[2]}
POINT_DATA {num_points}
SCALARS u float
LOOKUP_TABLE default
"""
        f.write(header.encode("utf-8"))
        f.write(u.astype('>f4').tobytes())  # '>f4' = big-endian float32


def run_gliodil(data_path, Nt, Nx, Ny, Nz, device, out_dir,
                trim_scale=1.5,
                num_epochs=5000, lr=1e-2, report_every=100,
                verbose=True, tend=50.0, lambda_pde=10, lambda_ic=100,
                matter_th=0.1):

    os.makedirs(out_dir, exist_ok=True)

    T_ig, u_ig, params_ig = get_initial_guess(data_path, Nx, Ny, Nz, trim_scale=trim_scale, Nt_ODIL=Nt, verbose=verbose)

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

    print(f"dx = {dx:.2f}mm, dy = {dy:.2f}mm, dz = {dz:.2f}mm")

    gm_ = torch.from_numpy(gm).to(device)
    wm_ = torch.from_numpy(wm).to(device)

    matter = get_matter_portions(gm, wm, threshold=matter_th, device=device)
    seg_ = torch.from_numpy(seg).to(device)

    mask_core = torch.where(seg_ == SEG_CODE.core, 1.0, 0.0)
    mask_edema = torch.where(seg_ == SEG_CODE.edema, 1.0, 0.0)

    # parameters
    tscale = tend / T_ig
    Dw  = params_ig['Dw']  * tscale
    Dg  = params_ig['Dg']  * tscale
    rho = params_ig['rho'] * tscale
    x0, y0, z0 = params_ig['x0'], params_ig['y0'], params_ig['z0']
    th_core = params_ig['th_hi']
    th_lo = params_ig['th_lo']
    th_hi = params_ig['th_hi']

    if verbose:
        print("Initial guess for parameters:")
        print(f"    (x0, y0, z0) = ({x0:.1f}, {y0:.1f}, {z0:.1f})")


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
        res_ic = u[0,:,:,:] - u0

        return lambda_ic * torch.mean(res_ic**2)

    def compute_data_loss(u, th_lo, th_hi):
        uend = u[-1,:,:,:]

        # map of the range of values uend should take.
        lower_vals = torch.where(seg_ == SEG_CODE.core, th_hi, torch.where(seg_ == SEG_CODE.edema, th_lo, 0.0))
        upper_vals = torch.where(seg_ == SEG_CODE.core, 1.0, torch.where(seg_ == SEG_CODE.edema, th_hi, th_lo))

        # residuals of values that are too low
        relu = torch.nn.functional.relu
        res_lo = relu(lower_vals - uend)
        res_hi = relu(uend - upper_vals)

        return torch.mean(res_lo + res_hi)

    def compute_csf_loss(u):
        mask = torch.where(gm_ + wm_ < matter_th, 1.0, 0.0)
        res = mask[None,:,:,:] * u
        return torch.mean(res**2)

    def params_bounds_loss(th_lo, th_hi):
        relu = torch.nn.functional.relu

        return \
            relu(0.20 - th_lo) + relu(th_lo - 0.5) + \
            relu(0.50 - th_hi) + relu(th_hi - 0.85)


    # initial guess
    u0 = torch.from_numpy(u_ig)
    depth = int(np.log(min([Nt, Nx, Ny, Nz])) / np.log(2))
    if verbose:
        print(f"Multigrid depth = {depth}")
    mg = MultigridField(u0, loc='nppp', depth=depth)
    mg.to(device)
    mg.set_requires_grad()

    params = torch.tensor([Dw, np.log(Dw/Dg), rho, x0, y0, z0, th_lo, th_hi], requires_grad=True)
    params.to(device)

    optim = torch.optim.Adam(mg.params() + [params], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=10, min_lr=1e-4)

    epochs = list(range(num_epochs))
    pde_losses = []
    data_losses = []
    ic_losses = []
    losses = []

    for epoch in epochs:
        optim.zero_grad()
        u = mg.get()
        Dw, log_R, rho, x0, y0, z0, th_lo, th_hi = params
        R = torch.exp(log_R)
        Dg = Dw / R
        ploss = compute_pde_loss(u, Dw=Dw, Dg=Dg, rho=rho)
        dloss = compute_data_loss(u, th_lo=th_lo, th_hi=th_hi)
        iloss = compute_ic_loss(u, x0, y0, z0)
        loss = ploss + iloss + dloss
        loss += params_bounds_loss(th_lo, th_hi)
        loss += compute_csf_loss(u)
        loss.backward()
        optim.step()

        l = loss.item()
        pde_losses.append(ploss.item())
        data_losses.append(dloss.item())
        ic_losses.append(iloss.item())
        losses.append(l)

        scheduler.step(l)

        if verbose and epoch % report_every == 0:
            params_str = ''.join(f"{v:.3f} " for v in params.detach().numpy())
            print(f"epoch {epoch:06d} loss {l:.4e}, params {params_str}")

    train_hist = {
        'epoch': epochs,
        'pde_loss': pde_losses,
        'data_loss': data_losses,
        'ic_loss': ic_losses,
        'loss': losses
    }

    pd.DataFrame(train_hist).to_csv(os.path.join(out_dir, 'train_history.csv'), index=False)

    u = mg.get().detach().cpu().numpy()
    uend = u[-1,:,:,:]

    for it in range(Nt):
        dump_vtk(u[it], dx, dy, dz, path=os.path.join(out_dir, f'seg_{it:04d}.vtk'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path to .nii files')
    parser.add_argument('--NtNxNyNz', type=int, nargs=4, default=[129, 64, 64, 64], help='odil grid size (Nt, Nx, Ny, Nz)')
    parser.add_argument('--out-dir', type=str, default='out_gliodil', help='output directory')
    args = parser.parse_args()

    Nt, Nx, Ny, Nz = args.NtNxNyNz
    out_dir = args.out_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_gliodil(data_path=args.data_path,
                Nt=Nt, Nx=Nx, Ny=Ny, Nz=Nz,
                device=device, out_dir=out_dir)

if __name__ == '__main__':
    main()
