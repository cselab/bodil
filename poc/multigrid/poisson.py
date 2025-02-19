#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch

def upscale_grid(u):
    dim = np.ndim(u)
    new_shape = tuple(2 * s - 1 for s in u.shape)
    uf = torch.zeros(new_shape, dtype=u.dtype)

    # Copy original points
    uf[tuple([slice(None, None, 2)] * dim)] = u

    # Interpolate along each axis
    for axis in range(dim):
        slices1 = tuple(slice(None, -1, 2) if i == axis else slice(None) for i in range(dim))
        slices2 = tuple(slice(1, None, 2) if i == axis else slice(None) for i in range(dim))
        uf[slices2] = (uf[slices1] + torch.roll(uf[slices1], shifts=-1, dims=axis)) / 2

    return uf


def create_mg(u, depth):
    shape = np.array((np.shape(u)), dtype=int)
    mg = [u.clone()]
    for d in range(depth):
        for s in shape:
            assert (s+1) % 2 == 0
            assert s > 1
        shape = (shape + 1) // 2
        mg.append(torch.zeros(tuple(shape), dtype=u.dtype, requires_grad=u.requires_grad, device=u.device))
    return mg


def mg_to_field(mg):
    depth = len(mg)
    u = mg[-1]
    for subu in mg[-2::-1]:
        u = upscale_grid(u) + subu
    return u


def ODIL_train(nx=129, num_epochs=100000, lr=1e-3, depth=1):
    L = 2 * np.pi
    dx = L / nx

    x = torch.linspace(0, L, nx)
    u = torch.zeros_like(x)

    mg = create_mg(u, depth)
    for i in range(len(mg)):
        mg[i].requires_grad=True

    epochs = list(range(num_epochs))
    losses = []
    optim = torch.optim.Adam(mg, lr=1e-3)

    def odil_loss(u):
        # bulk
        d2udx2 = (2 * u[1:-1] - u[:-2] - u[2:]) / (dx**2)
        residuals = d2udx2
        loss = torch.mean(residuals**2)

        # BC
        dudxl = (u[1] - u[0]) / dx
        dudxr = (u[-1] - u[-2]) / dx
        loss += (dudxl - 1)**2
        loss += (dudxr + 1)**2
        loss += u[0]**2

        return loss

    for epoch in epochs:
        optim.zero_grad()
        u = mg_to_field(mg)
        loss = odil_loss(u)
        loss.backward()
        optim.step()
        losses.append(loss.item())

        if epoch % 10000 == 0:
            print(f"epoch {epoch:05d} loss {loss.item():.4e}")

    return epochs, losses


def main():

    fig, ax = plt.subplots()

    for depth in range(8):
        print(f"depth {depth}")
        epochs, losses = ODIL_train(depth=depth)
        ax.plot(epochs, losses, label=f"depth = {depth}")
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1,None)
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
