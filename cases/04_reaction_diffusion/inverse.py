#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from uq_odil.multigrid import MultigridField

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-dir", type=str, default="out_forward", help="output directory of forward.py")
    parser.add_argument("--out-dir", type=str, default="out_inverse", help="output directory")
    parser.add_argument("--initial-pos", type=float, nargs=2, default=[2/3, 1/3], help="position of initial tumor")
    parser.add_argument("--dump-snapshots", action='store_true', default=False, help="if set, dump images of field.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float32)

    num_epochs = 10000
    report_every = 100
    lr = 1e-4

    forward_dir = args.forward_dir
    out_dir = args.out_dir
    x0, y0 = args.initial_pos

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(forward_dir, "diff_field.npy"), "rb") as f:
        diff_field = np.load(f)

    with open(os.path.join(forward_dir, "u_final.npy"), "rb") as f:
        u_final = torch.from_numpy(np.load(f)).to(device)

    ny, nx = u_final.shape
    assert diff_field.shape[0] == ny
    assert diff_field.shape[1] == nx

    rho = 8
    L = 1.0
    tend = 1.0
    nt = 129
    dt = tend / nt

    x = np.linspace(0, L, nx, endpoint=False)
    y = np.linspace(0, L, ny, endpoint=False)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y)
    X = torch.from_numpy(X).to(device)
    Y = torch.from_numpy(Y).to(device)

    Dm0 = torch.from_numpy((diff_field + np.roll(diff_field, shift=+1, axis=1)) / 2).to(device)[:,:,None]
    Dp0 = torch.from_numpy((diff_field + np.roll(diff_field, shift=-1, axis=1)) / 2).to(device)[:,:,None]
    D0m = torch.from_numpy((diff_field + np.roll(diff_field, shift=+1, axis=0)) / 2).to(device)[:,:,None]
    D0p = torch.from_numpy((diff_field + np.roll(diff_field, shift=-1, axis=0)) / 2).to(device)[:,:,None]

    mg = MultigridField(torch.zeros((ny, nx, nt)), loc='ppn', depth=6)
    mg.set_requires_grad()
    mg.to(device)

    def pde_loss(u):
        um0 = torch.roll(u, shifts=+1, dims=1)
        up0 = torch.roll(u, shifts=-1, dims=1)
        u0m = torch.roll(u, shifts=+1, dims=0)
        u0p = torch.roll(u, shifts=-1, dims=0)

        A  =  (Dp0 * (up0 - u) - Dm0 * (u - um0)) / dx**2 \
            + (D0p * (u0p - u) - D0m * (u - u0m)) / dy**2
        B = rho * u * (1 - u)

        rhs = (A[:,:,1:] + A[:,:,:-1] + B[:,:,1:] + B[:,:,:-1]) / 2
        dudt = torch.diff(u, dim=-1) / dt

        residuals = dudt - rhs

        return torch.mean(residuals**2)

    def data_loss(u):
        residuals = u[:,:,-1] - u_final
        return 10 * torch.mean(residuals**2)

    def init_loss(u, x0, y0, R0):
        u0 = u[:,:,0]
        R = (1 + torch.tanh(R0)) / 2 * L/16
        u0_guess = torch.exp(-((X-x0)**2 + (Y-y0)**2) / (2 * 0.0001 + (R**2)))
        residuals = u0 - u0_guess
        return 5 * torch.mean(residuals**2)


    R0 = L/32
    init_params = torch.tensor([R0]).to(device)
    init_params.requires_grad = True

    optim = torch.optim.Adam([init_params] + mg.params(), lr=lr)

    epochs = list(range(num_epochs))
    pde_losses = []
    data_losses = []
    losses = []

    for epoch in epochs:
        optim.zero_grad()
        u = mg.get()
        ploss = pde_loss(u)
        dloss = data_loss(u)
        iloss = init_loss(u, x0, y0, *init_params)
        loss = ploss + dloss + iloss
        loss.backward()
        optim.step()

        l = loss.item()
        pde_losses.append(ploss.item())
        data_losses.append(dloss.item())
        losses.append(l)

        if epoch % report_every == 0:
            print(f"epoch {epoch:06d} loss {l:.4e}")

    train_hist = {
        'epoch': epochs,
        'pde_loss': pde_losses,
        'data_loss': data_losses,
        'loss': losses
    }

    pd.DataFrame(train_hist).to_csv(os.path.join(out_dir, 'train_history.csv'), index=False)

    # save snapshots
    if args.dump_snapshots:
        for i in range(nt):
            u = mg.get()
            u_ = u[:,:,i].detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(u_, origin='lower', extent=[0, L, 0, L], vmin=0, vmax=1)
            plt.savefig(os.path.join(out_dir, f"u-{i:06d}.png"))
            plt.close(fig)



if __name__ == '__main__':
    main()
