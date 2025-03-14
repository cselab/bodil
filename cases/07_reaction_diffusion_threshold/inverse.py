#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from uq_odil.multigrid import MultigridField

def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))

def run(forward_dir, out_dir, initial_pos, dump_snapshots, threshold, sigma_data, device):
    torch.set_default_dtype(torch.float32)

    num_epochs = 20000
    report_every = 2000
    lr = 1e-3

    x0, y0 = initial_pos

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(forward_dir, "diff_field.npy"), "rb") as f:
        diff_field = np.load(f)

    with open(os.path.join(forward_dir, "ut_final.npy"), "rb") as f:
        ut_final = torch.from_numpy(np.load(f)).to(device)

    ny, nx = ut_final.shape
    assert diff_field.shape[0] == ny
    assert diff_field.shape[1] == nx

    rho = 8
    L = 1.0
    tend = 0.5
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

    # initial guess
    u0 = torch.full((ny, nx, nt), threshold)
    #u0[:,:,-1] = ut_final

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

        return 1e3 * torch.mean(residuals**2)

    def data_loss(u):
        alphas = torch.sigmoid((u[:,:,-1] - threshold))
        neg_loss = ut_final * torch.log(alphas) + (1-ut_final) * torch.log(1-alphas)
        return -torch.mean(neg_loss)

    def init_loss(u, x0, y0, R0):
        u0 = u[:,:,0]
        u0_guess = torch.exp(-((X-x0)**2 + (Y-y0)**2) / (2 * R0**2))
        residuals = u0 - u0_guess
        return 1e3 * torch.mean(residuals**2)


    R0 = L/32

    optim = torch.optim.Adam(mg.params(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=10, min_lr=1e-4)

    epochs = list(range(num_epochs))
    pde_losses = []
    data_losses = []
    losses = []

    for epoch in epochs:
        optim.zero_grad()
        u = mg.get()
        ploss = pde_loss(u)
        dloss = data_loss(u)
        iloss = init_loss(u, x0, y0, R0)
        loss = ploss + dloss + iloss
        loss.backward()
        optim.step()

        l = loss.item()
        pde_losses.append(ploss.item())
        data_losses.append(dloss.item())
        losses.append(l)

        scheduler.step(l)

        if epoch % report_every == 0:
            print(f"epoch {epoch:06d} loss {l:.4e} lr {scheduler.get_last_lr()}")

    train_hist = {
        'epoch': epochs,
        'pde_loss': pde_losses,
        'data_loss': data_losses,
        'loss': losses
    }

    pd.DataFrame(train_hist).to_csv(os.path.join(out_dir, 'train_history.csv'), index=False)

    u = mg.get()

    with open(os.path.join(out_dir, "u_final.npy"), "wb") as f:
        np.save(f, u[:,:,-1].detach().cpu().numpy())

    # save snapshots
    if dump_snapshots:
        for i in range(nt):
            u_ = u[:,:,i].detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(u_, origin='lower', extent=[0, L, 0, L], vmin=0, vmax=1)
            plt.savefig(os.path.join(out_dir, f"u-{i:06d}.png"))
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-dir", type=str, default="out_forward", help="output directory of forward.py")
    parser.add_argument("--out-dir", type=str, default="out_inverse", help="output directory")
    parser.add_argument("--initial-pos", type=float, nargs=2, default=[2/3, 1/3], help="position of initial tumor")
    parser.add_argument("--dump-snapshots", action='store_true', default=False, help="if set, dump images of field.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Measurement threshold.")
    parser.add_argument("--sigma-data", type=float, default=0.001, help="Data uncertainty parameter.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run(forward_dir=args.forward_dir,
        out_dir=args.out_dir,
        initial_pos=args.initial_pos,
        dump_snapshots=args.dump_snapshots,
        threshold=args.threshold,
        sigma_data=args.sigma_data,
        device=device)


if __name__ == '__main__':
    main()
