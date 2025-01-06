#!/usr/bin/env python3

import numpy as np
import torch
from torch.optim import Adam

def main():

    """
    Solve the advection equation in 1D
    du/dt + a * du/dx = 0
    with periodic B.C.
    and a sine initial conditions
    """

    num_epochs = 200000
    lr = 0.001

    a = 1.0

    nx = 128
    nt = 64

    L = 1.0
    T = 1.0

    x = np.linspace(0, L, nx, endpoint=False)
    t = np.linspace(0, T, nt + 1, endpoint=True)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    x = torch.from_numpy(x)

    u0 = torch.sin(x * 2 * np.pi / L)
    u = torch.zeros((nx, nt), requires_grad=True)

    optim = Adam([u], lr=lr)

    def compute_loss(u):
        dudt = torch.diff(u, dim=1) / dt
        dudx = (torch.roll(u, shifts=-1, dims=0) - torch.roll(u, shifts=+1, dims=0)) / (2 * dx)

        pde_residuals = dudt + a * dudx[:,1:] # forward euler
        bc_residuals = u[:,0] - u0
        loss = torch.sum(pde_residuals**2) + torch.sum(bc_residuals**2)
        return loss

    epochs = list(range(num_epochs))
    losses = []
    for epoch in epochs:
        optim.zero_grad()
        loss = compute_loss(u)
        loss.backward()
        optim.step()
        l = float(loss.detach().cpu().float())
        if epoch % 10000 == 0:
            print(f"epoch {epoch:06d}, loss {l:.6e}")

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
    ax.set_yscale('log')
    plt.show()





if __name__ == '__main__':
    main()
