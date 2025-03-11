#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from uq_odil.multigrid import MultigridField
from random_field import generate_random_field

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="out_forward", help="output directory")
    parser.add_argument("--dump-snapshots", action='store_true', default=False, help="if set, dump images of field.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Measurement threshold.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 20000
    report_every = 2000
    lr = 1e-3

    threshold = args.threshold
    out_dir = args.out_dir
    dump = args.dump_snapshots
    os.makedirs(out_dir, exist_ok=True)
    Dw = 0.005
    Dg = 0.1
    rho = 8
    L = 1.0
    nx = 64
    ny = 64
    tend = 0.5
    nt = 129
    dt = tend / nt
    rng = np.random.default_rng(seed=123456)

    x = np.linspace(0, L, nx, endpoint=False)
    y = np.linspace(0, L, ny, endpoint=False)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    diff_field = generate_random_field(nx, ny, smoothness=nx/8, rng=rng).T
    diff_field = np.where(diff_field > 0, Dw, Dg)

    if dump:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(diff_field, origin='lower', extent=[0, L, 0, L], vmin=Dw, vmax=Dg)
        plt.savefig(os.path.join(out_dir, "diff_field.png"))
        plt.close(fig)

    with open(os.path.join(out_dir, "diff_field.npy"), "wb") as f:
        np.save(f, diff_field)

    X, Y = np.meshgrid(x, y)

    # initial conditions
    R0 = L/32
    x0 = 2 * L / 3
    y0 = L / 3
    u_init = torch.from_numpy(np.exp(-((X-x0)**2 + (Y-y0)**2) / (2 * R0**2)))

    u0 = torch.zeros((ny, nx, nt))
    u0[:,:,0] = u_init

    X = torch.from_numpy(X).to(device)
    Y = torch.from_numpy(Y).to(device)

    Dm0 = torch.from_numpy((diff_field + np.roll(diff_field, shift=+1, axis=1)) / 2).to(device)[:,:,None]
    Dp0 = torch.from_numpy((diff_field + np.roll(diff_field, shift=-1, axis=1)) / 2).to(device)[:,:,None]
    D0m = torch.from_numpy((diff_field + np.roll(diff_field, shift=+1, axis=0)) / 2).to(device)[:,:,None]
    D0p = torch.from_numpy((diff_field + np.roll(diff_field, shift=-1, axis=0)) / 2).to(device)[:,:,None]

    mg = MultigridField(u0, loc='ppn', depth=6)
    mg.to(device)
    mg.set_requires_grad()

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

    def init_loss(u, x0, y0, R0):
        u0 = u[:,:,0]
        u0_guess = torch.exp(-((X-x0)**2 + (Y-y0)**2) / (2 * R0**2))
        residuals = u0 - u0_guess
        return 5 * torch.mean(residuals**2)

    optim = torch.optim.Adam(mg.params(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=10, min_lr=1e-4)

    epochs = list(range(num_epochs))
    pde_losses = []
    losses = []

    for epoch in epochs:
        optim.zero_grad()
        u = mg.get()
        ploss = pde_loss(u)
        iloss = init_loss(u, x0, y0, R0)
        loss = ploss + iloss
        loss.backward()
        optim.step()

        l = loss.item()
        pde_losses.append(ploss.item())
        losses.append(l)

        scheduler.step(l)

        if epoch % report_every == 0:
            print(f"epoch {epoch:06d} loss {l:.4e} lr {scheduler.get_last_lr()}")

    train_hist = {
        'epoch': epochs,
        'pde_loss': pde_losses,
        'loss': losses
    }
    pd.DataFrame(train_hist).to_csv(os.path.join(out_dir, 'train_history.csv'), index=False)

    u = mg.get()

    uf = u[:,:,-1].detach().cpu().numpy()
    ut = np.where(uf > threshold, 1.0, 0.0)
    with open(os.path.join(out_dir, "ut_final.npy"), "wb") as f:
        np.save(f, ut)

    with open(os.path.join(out_dir, "u_final.npy"), "wb") as f:
        np.save(f, uf)

    # save snapshots
    if dump:
        for i in range(nt):
            u_ = u[:,:,i].detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(u_, origin='lower', extent=[0, L, 0, L], vmin=0, vmax=1)
            plt.savefig(os.path.join(out_dir, f"u-{i:06d}.png"))
            plt.close(fig)


if __name__ == '__main__':
    main()
