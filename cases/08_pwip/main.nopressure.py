#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy
import torch

def find_A0(data_u, data_A):
    i, j = np.unravel_index(np.argmin(data_u**2), data_u.shape)
    A0 = data_A[i, j]
    return A0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='path to .mat file containing the data')
    parser.add_argument('--fit-Kr', action='store_true', default=False)
    parser.add_argument('--out-dir', type=str, default='out', help='path to output directory')
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)

    out_dir = args.out_dir
    data_path = args.data
    fit_Kr = args.fit_Kr

    os.makedirs(out_dir, exist_ok=True)

    num_epochs = 5000
    report_every = 250
    lr = 5e-3
    lambda_PDE = 100
    lambda_data = 1

    data = scipy.io.loadmat(data_path)
    data_A, data_u, data_P, data_t, data_x = data['matrix_var'].T
    nt, nx = data_A.shape

    # rescale time and space to get mm and ms
    l_scale = 1e-3 # 1mm
    t_scale = 2e-3 # 2ms
    m_scale = 1e-6 # mg
    v_scale = l_scale / t_scale
    p_scale = m_scale / (l_scale * t_scale**2)
    k_scale = l_scale**3 * t_scale**2 / m_scale

    data_t /= t_scale
    data_x /= l_scale
    data_A /= l_scale**2
    data_u /= v_scale
    data_P /= p_scale

    print("u", np.mean(data_u),  np.ptp(data_u))
    print("A", np.mean(data_A),  np.ptp(data_A))


    A0 = find_A0(data_u, data_A)
    Kr = 0
    rho = 1000.689275457646

    rho *= l_scale**3 / m_scale

    dt = data_t[1,0] - data_t[0,0]
    dx = data_x[0,1] - data_x[0,0]
    print(f"dx = {dx} mm, dt = {dt} ms")
    print(f"A0 = {A0} mm**2")

    # transfer data to pytorch
    data_A_ = torch.from_numpy(data_A)
    data_u_ = torch.from_numpy(data_u)

    def pde_loss(kp, u, A, A0, Kr):
        dAdt = torch.diff((A[:,:-1] + A[:,1:]) / 2, dim=0) / dt
        dAdx = torch.diff((A[:-1,:] + A[1:,:]) / 2, dim=1) / dx

        dudt = torch.diff((u[:,:-1] + u[:,1:]) / 2, dim=0) / dt
        dudx = torch.diff((u[:-1,:] + u[1:,:]) / 2, dim=1) / dx

        dkpdx = torch.diff(kp) / dx

        umm = (u[:-1,:-1] + u[1:,:-1] + u[:-1,1:] + u[1:,1:]) / 4
        Amm = (A[:-1,:-1] + A[1:,:-1] + A[:-1,1:] + A[1:,1:]) / 4
        kpm = (kp[:-1] + kp[1:]) / 2

        res0 = dAdt + A0 * dudx
        res1 = dudt + (dAdx - (Amm - A0) * dkpdx[None,:] / kpm[None,:]) / (rho * kpm[None,:]) + Kr * umm

        return lambda_PDE * (100 * torch.mean(res0**2) + torch.mean(res1**2))

    def data_loss(u, A):
        res_A = data_A_ - A
        res_u = data_u_ - u

        return lambda_data * (torch.mean(res_A**2) + torch.mean(res_u**2))

    # unknowns
    u = data_u_.clone().detach()
    u.requires_grad = True
    kp0 = 2e-9 / k_scale
    kp = torch.full((nx,), fill_value=float(kp0), requires_grad=True)

    if fit_Kr:
        Kr = torch.tensor([0.0], requires_grad=True)

    A = data_A_.clone()
    A.requires_grad = True

    unknowns = [kp, u, A]
    if fit_Kr:
        unknowns.append(Kr)

    optim = torch.optim.Adam(unknowns, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=300, gamma=0.7)

    epochs = list(range(num_epochs))
    pde_losses = []
    data_losses = []
    losses = []

    for epoch in epochs:
        optim.zero_grad()
        ploss = pde_loss(kp, u, A, A0, Kr)
        dloss = data_loss(u, A)
        loss = ploss + dloss
        loss.backward()
        optim.step()

        l = loss.item()
        pde_losses.append(ploss.item())
        data_losses.append(dloss.item())
        losses.append(l)

        scheduler.step()

        if epoch % report_every == 0:
            if fit_Kr:
                print(f"epoch {epoch:06d} loss {l:.4e}, Kr = {Kr.item()}")
            else:
                print(f"epoch {epoch:06d} loss {l:.4e}")

    train_hist = {
        'epoch': epochs,
        'pde_loss': pde_losses,
        'data_loss': data_losses,
        'loss': losses
    }

    pd.DataFrame(train_hist).to_csv(os.path.join(out_dir, 'train_history.csv'), index=False)

    x = data_x[0,:]
    t = data_t[:,0]

    kp_ = kp.detach().numpy() * k_scale

    fig, ax = plt.subplots()
    ax.plot(x, kp_)
    ax.set_xlabel(r"$x$ (mm)")
    ax.set_ylabel(r"$k_p$ (m$^3$ s$^2$ / kg)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kp.pdf"))
    plt.close()

    t0 = data_t[0, 0]
    t1 = data_t[-1, 0]
    x0 = data_x[0, 0]
    x1 = data_x[0, -1]

    u = u.detach().numpy() * l_scale / t_scale * 100 # m/s -> cm/s
    du = np.abs(u - data_u * l_scale / t_scale * 100)

    fig, ax = plt.subplots()
    im = ax.imshow(u.T, extent=(t0, t1, x0, x1), origin='lower', aspect='auto', cmap='jet')
    ax.set_xlabel(r"$t$ (ms)")
    ax.set_ylabel(r"$x$ (mm)")
    fig.colorbar(im, ax=ax, label=r"$u$ (cm/s)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "u.pdf"))
    plt.close()

    fig, ax = plt.subplots()
    im = ax.imshow(du.T, extent=(t0, t1, x0, x1), origin='lower', aspect='auto', cmap='jet')
    ax.set_xlabel(r"$t$ (ms)")
    ax.set_ylabel(r"$x$ (mm)")
    fig.colorbar(im, ax=ax, label=r"$|\delta u|$ (cm/s)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "du.pdf"))
    plt.close()

    A = A.detach().numpy() * l_scale**2
    dA = np.abs(A - data_A * l_scale**2)

    A *= 1e6 # m^2 -> mm^2
    dA *= 1e6 # m^2 -> mm^2

    fig, ax = plt.subplots()
    im = ax.imshow(A.T, extent=(t0, t1, x[0], x[-1]), origin='lower', aspect='auto', cmap='jet')
    ax.set_xlabel(r"$t$ (ms)")
    ax.set_ylabel(r"$x$ (mm)")
    fig.colorbar(im, ax=ax, label=r"$A$ (mm$^2$)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "A.pdf"))
    plt.close()

    fig, ax = plt.subplots()
    im = ax.imshow(dA.T, extent=(t0, t1, x[0], x[-1]), origin='lower', aspect='auto', cmap='jet')
    ax.set_xlabel(r"$t$ (ms)")
    ax.set_ylabel(r"$x$ (mm)")
    fig.colorbar(im, ax=ax, label=r"$|\delta A|$ (mm$^2$)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dA.pdf"))
    plt.close()

if __name__ == '__main__':
    main()
