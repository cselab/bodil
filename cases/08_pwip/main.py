#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='path to .mat file containing the data')
    parser.add_argument('--out-dir', type=str, default='out', help='path to output directory')
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)

    out_dir = args.out_dir
    data_path = args.data

    os.makedirs(out_dir, exist_ok=True)

    num_epochs = 2000
    report_every = 100
    lr = 1e-3
    lambda_PDE = 1000

    data = scipy.io.loadmat(data_path)
    data_A, data_u, data_P, data_t, data_x = data['matrix_var'].T
    nt, nx = data_A.shape

    # rescale time and space to get mm and ms
    l_scale = 1e-3 # 1mm
    t_scale = 1e-3 # 1ms
    m_scale = 1e-6 # mg

    data_t /= t_scale
    data_x /= l_scale
    data_A /= l_scale**2
    data_u *= t_scale / l_scale
    data_P *= t_scale**2 * l_scale / m_scale

    rho = 1000 * l_scale**3 / m_scale

    dt = data_t[1,0] - data_t[0,0]
    dx = data_x[0,1] - data_x[0,0]
    print(f"dx = {dx} mm, dt = {dt} ms")

    # transfer data to pytorch
    data_A_ = torch.from_numpy(data_A)
    data_u_ = torch.from_numpy(data_u)
    data_P_ = torch.from_numpy(data_P)

    def pde_loss(kp, u, P, A0, Kr):
        dPdt = torch.diff((P[:,:-1] + P[:,1:]) / 2, dim=0) / dt
        dPdx = torch.diff((P[:-1,:] + P[1:,:]) / 2, dim=1) / dx

        dudt = torch.diff((u[:,:-1] + u[:,1:]) / 2, dim=0) / dt
        dudx = torch.diff((u[:-1,:] + u[1:,:]) / 2, dim=1) / dx

        um = (u[:-1,:-1] + u[1:,:-1] + u[:-1,1:] + u[1:,1:]) / 4

        res0 = kp[None, :] * dPdt + A0 * dudx
        res1 = dudt + dPdx / rho + Kr * um

        return lambda_PDE * (torch.mean(res0**2) + torch.mean(res1**2))

    def data_loss(kp, u, P, A0):
        data_Am = (data_A_[:-1,:-1] + data_A_[1:,:-1] + data_A_[:-1,1:] + data_A_[1:,1:]) / 4
        Pm = (P[:-1,:-1] + P[1:,:-1] + P[:-1,1:] + P[1:,1:]) / 4
        res_A = data_Am - A0 - kp[None,:] * Pm
        res_P = data_P_ - P
        res_u = data_u_ - u

        return torch.mean(res_A**2) + torch.mean(res_P**2) + torch.mean(res_u**2)

    # unknowns
    u = data_u_.detach()
    u.requires_grad = True

    P = data_P_.detach()
    P.requires_grad = True

    A0 = torch.tensor(np.median(data_A))
    A0.requires_grad = True

    Kr = torch.tensor(0.0)
    Kr.requires_grad = True

    kp = torch.ones(nx-1, requires_grad=True)

    optim = torch.optim.Adam([kp, u, P, A0, Kr], lr=lr)

    epochs = list(range(num_epochs))
    pde_losses = []
    data_losses = []
    losses = []

    for epoch in epochs:
        optim.zero_grad()
        ploss = pde_loss(kp, u, P, A0, Kr)
        dloss = data_loss(kp, u, P, A0)
        loss = ploss + dloss
        loss.backward()
        optim.step()

        l = loss.item()
        pde_losses.append(ploss.item())
        data_losses.append(dloss.item())
        losses.append(l)

        if epoch % report_every == 0:
            print(f"epoch {epoch:06d} loss {l:.4e} A0 {A0.item()} Kr {Kr.item()}")

    train_hist = {
        'epoch': epochs,
        'pde_loss': pde_losses,
        'data_loss': data_losses,
        'loss': losses
    }

    pd.DataFrame(train_hist).to_csv(os.path.join(out_dir, 'train_history.csv'), index=False)

    x = data_x[0,:]
    xm = (x[:-1] + x[1:]) / 2
    t = data_t[:,0]

    kp_ = kp.detach().numpy() * l_scale**3 * t_scale**2 / m_scale

    fig, ax = plt.subplots()
    ax.plot(xm, kp_)
    ax.set_xlabel(r"$x$ (mm)")
    ax.set_ylabel(r"$k_p$ (m$^3$ s$^2$ / kg)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kp.pdf"))
    plt.close()

    inlet_P = P[:,0].detach().numpy() * m_scale / l_scale / t_scale**2

    fig, ax = plt.subplots()
    ax.plot(t, inlet_P)
    ax.set_xlabel(r"$t$ (ms)")
    ax.set_ylabel(r"$P$ (Pa)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "inlet_P.pdf"))
    plt.close()

    t0 = data_t[0, 0]
    t1 = data_t[-1, 0]
    x0 = data_x[0, 0]
    x1 = data_x[0, -1]

    u = u.detach().numpy() * l_scale / t_scale * 100 # m/s -> cm/s

    fig, ax = plt.subplots()
    im = ax.imshow(u.T, extent=(t0, t1, x0, x1), origin='lower', aspect='auto', cmap='jet')
    ax.set_xlabel(r"$t$ (ms)")
    ax.set_ylabel(r"$x$ (mm)")
    fig.colorbar(im, ax=ax, label=r"$u$ (cm/s)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "u.pdf"))
    plt.close()

    P = P.detach().numpy() * m_scale / l_scale / t_scale**2
    fig, ax = plt.subplots()
    im = ax.imshow(P.T, extent=(t0, t1, x0, x1), origin='lower', aspect='auto', cmap='jet')
    ax.set_xlabel(r"$t$ (ms)")
    ax.set_ylabel(r"$x$ (mm)")
    fig.colorbar(im, ax=ax, label=r"$P$ (Pa)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "P.pdf"))
    plt.close()

    Pm = (P[:,1:] + P[:,:-1]) / 2
    A = A0.item() * l_scale**2 + kp_ * Pm
    A *= 1e6 # m^2 -> mm^2

    fig, ax = plt.subplots()
    im = ax.imshow(A.T, extent=(t0, t1, xm[0], xm[-1]), origin='lower', aspect='auto', cmap='jet')
    ax.set_xlabel(r"$t$ (ms)")
    ax.set_ylabel(r"$x$ (mm)")
    fig.colorbar(im, ax=ax, label=r"$A$ (mm$^2$)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "A.pdf"))
    plt.close()

if __name__ == '__main__':
    main()
