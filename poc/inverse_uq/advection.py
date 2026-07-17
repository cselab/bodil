#!/usr/bin/env python3

import numpy as np
import torch
from torch.optim import Adam

def generate_data(num_data, L, T, a, rng, sigma=0.1):
    """
    Parameters:
        num_data: number of samples to generate
        L: domain size
        T: end time
        a: velocity
        rng: numpy random number generator
        sigma: noise level
    """
    x = rng.uniform(0, L, num_data)
    t = rng.uniform(0, T, num_data)
    u = np.sin((x - a * t) * 2 * np.pi / L) + rng.normal(0, sigma, num_data)
    return x, t, u


def main():
    """
    Solve the advection equation in 1D
    du/dt + a * du/dx = 0
    with periodic B.C. and a sine initial conditions.

    The parameter a is unknown and we find it through B-ODIL.
    """
    L = 1.0
    T = 1.0

    seed = 2349873
    num_epochs = 100000
    lr = 1e-3
    num_data = 20
    lambda_data = 5.0
    a0 = 1.0
    rng = np.random.default_rng(seed=seed)

    xd, td, ud = generate_data(num_data, L, T, a=1.0, rng=rng)

    nx = 128
    nt = 64

    x = np.linspace(0, L, nx, endpoint=False)
    t = np.linspace(0, T, nt + 1, endpoint=True)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    x = torch.from_numpy(x)

    xd_ids = torch.from_numpy((xd / dx).astype(int))
    td_ids = torch.from_numpy((td / dt).astype(int))
    ud = torch.from_numpy(ud)

    u0 = torch.sin(x * 2 * np.pi / L)
    u = torch.zeros((nx, nt), requires_grad=True)
    a = torch.tensor(a0, requires_grad=True)
    sigma = torch.tensor(1.0, requires_grad=True)


    optim = Adam([u, a, sigma], lr=lr)

    def compute_loss():
        dudt = torch.diff(u, dim=1) / dt
        dudx = (torch.roll(u, shifts=-1, dims=0) - u) / dx

        pde_residuals = dudt + a * dudx[:,1:] # forward euler
        ic_residuals = u[:,0] - u0
        data_residuals = u[xd_ids,td_ids] - ud
        s2pde = sigma**2 / np.sqrt(nx * nt)
        s2ic = sigma**2 / np.sqrt(nx)
        s2data = sigma**2 / np.sqrt(num_data) / lambda_data
        loss = torch.sum(pde_residuals**2) / (2 * s2pde) + nx * nt * torch.log(s2pde) / 2
        loss += torch.sum(ic_residuals**2) / (2 * s2ic**2) + nx * torch.log(s2ic) / 2
        loss += torch.sum(data_residuals**2) / (2 * s2data) + num_data * torch.log(s2data) / 2
        return loss

    epochs = list(range(num_epochs))
    losses = []
    for epoch in epochs:
        optim.zero_grad()
        loss = compute_loss()
        loss.backward()
        optim.step()
        l = float(loss.detach().cpu().float())
        a_ = float(a.detach().cpu().float())
        sigma2 = float(sigma.detach().cpu().float())**2
        if epoch % 1000 == 0:
            print(f"epoch {epoch:06d}, loss {l:.6e}, a {a_:.6f}, sigma^2 {sigma2:.4e}")

        losses.append(l)


    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(u.detach().numpy().T,
              origin='lower')
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    plt.show()

    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(epochs, losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    #ax.set_yscale('log')
    plt.show()





if __name__ == '__main__':
    main()
