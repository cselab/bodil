#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim import LBFGS

from uq_odil.HMC import HMC

def exact_solution(nx, nt, D, L, T):
    x = np.linspace(0, L, nx, endpoint=False)
    t = np.linspace(0, T, nt+1, endpoint=True)
    k = 2 * np.pi / L
    lam = k**2 * D
    u = np.exp(-lam * t[None,:]) * np.cos(k * x[:,None])
    return x, t, u

def generate_data(num_data, nx, nt, D, L, T, rng, sigma):
    x = np.linspace(0, L, nx, endpoint=False)
    t = np.linspace(0, T, nt+1, endpoint=True)

    xd = rng.choice(x, num_data, replace=True)
    td = rng.choice(t, num_data, replace=True)

    k = 2 * np.pi / L
    lam = k**2 * D
    ud = np.exp(-lam * td) * np.cos(k * xd) + rng.normal(0, sigma, len(xd))

    return xd, td, ud


def main():
    seed = 2349873
    num_epochs = 500
    num_samples = 5000
    lr = 5e-4
    sigma_data = 0.1
    sigma_pde = 0.1
    rng = np.random.default_rng(seed=seed)
    num_data = 200

    T = 1.0
    D = 0.1
    L = 1.0

    nx = 16
    nt = 63

    x, t, uexact = exact_solution(nx=nx, nt=nt, D=D, L=L, T=T)
    xd, td, ud = generate_data(num_data=num_data, nx=nx, nt=nt, D=D, L=L, T=T, rng=rng, sigma=sigma_data)

    dx = x[1] - x[0]
    dt = t[1] - t[0]

    u_ = torch.zeros(nx * (nt + 1), requires_grad=True)
    td_ids = (td / dt).astype(int)
    xd_ids = (xd / dx).astype(int)

    td_ids_ = torch.from_numpy(td_ids)
    xd_ids_ = torch.from_numpy(xd_ids)
    ud_ = torch.from_numpy(ud)

    optim = LBFGS([u_], lr=lr)

    def neg_log_posterior(u_):
        u = u_.reshape((nx, nt+1))
        dudt = torch.diff(u, axis=1) / dt
        um = (u[:,:-1] + u[:,1:]) / 2
        d2udx2 = (torch.roll(um, shifts=-1, dims=0) - 2 * um + torch.roll(um, shifts=1, dims=0)) / (dx**2)

        pde_res = dudt - D * d2udx2
        data_res = u[xd_ids_, td_ids_] - ud_

        log_like  = torch.sum(-pde_res**2 / (2 * sigma_pde**2)) - (nx * nt)/2 * np.log(2 * np.pi * sigma_pde**2)
        log_like += torch.sum(-data_res**2 / (2 * sigma_data**2)) - num_data/2 * np.log(2 * np.pi * sigma_data**2)

        return -log_like

    epochs = list(range(num_epochs))
    losses = []
    for epoch in epochs:
        def closure():
            optim.zero_grad()
            loss = neg_log_posterior(u_)
            loss.backward()
            return loss

        optim.step(closure)
        if epoch % 100 == 0:
            l = neg_log_posterior(u_).detach().item()
            print(f"epoch {epoch:06d}, loss {l:.6e}")

        losses.append(l)


    u = u_.detach().numpy()
    u = u.reshape((nx, (nt+1)))

    if 1:
        fig, ax = plt.subplots()
        im = ax.imshow(u.T, origin='lower',
                       cmap="seismic",
                       vmin=-1, vmax=1,
                       aspect=2 * nx/(nt+1))
        fig.colorbar(im, ax=ax)
        ax.plot(xd_ids, td_ids, '.k')
        ax.set(xticks=np.linspace(0, nx, 3), xticklabels=np.linspace(0, L, 3),
               yticks=np.linspace(nt+1, 0, 5), yticklabels=np.linspace(0, T, 5))
        ax.set_xlabel(r'$x/L$')
        ax.set_ylabel(r'$t/T$')
        plt.show()
        plt.close()


    # HMC sampling
    hmc = HMC([u_], dt=3.5e-4, L=10, M=1)

    def closure():
        hmc.zero_grad()
        U = neg_log_posterior(u_)
        U.backward()
        return U

    samples = []
    num_accepted = 0
    Umap = None
    umap = None
    for k in range(num_samples):
        y_, H_, U_, accepted = hmc.step(closure)
        samples.append(y_[0].detach().numpy())
        num_accepted += accepted
        if Umap is None or Umap > U_:
            umap = y_[0].detach().numpy()
            Umap = U_

    print(f"accptance rate: {num_accepted/num_samples}")
    samples = np.array(samples)

    ushape = (nx, nt+1)
    umean = np.mean(samples, axis=0).reshape(ushape)
    ulo = np.quantile(samples, q=0.05, axis=0).reshape(ushape)
    uhi = np.quantile(samples, q=0.95, axis=0).reshape(ushape)

    for tid in [0, 4, 8, 16, 32]:
        fig, ax = plt.subplots()
        ax.fill_between(x, ulo[:,tid], uhi[:,tid], lw=0, alpha=0.2, color='r', label='5-95% quantiles of posterior')
        ax.plot(x, umean[:,tid], '-r', label='mean')
        ax.plot(x, uexact[:,tid], '--k', label='exact')
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u(t={:.2f}, x)$".format(t[tid]))
        ax.set_xlim(0, L)
        ax.set_ylim(-1.2, 1.2)
        ax.legend(frameon=False)
        plt.show()


if __name__ == '__main__':
    main()
