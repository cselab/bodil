#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch


def upscale_grid(u):
    dim = np.ndim(u)
    new_shape = tuple(2 * s - 1 for s in u.shape)
    uf = torch.zeros(new_shape, dtype=u.dtype)
    uf[tuple([slice(None, None, 2)] * dim)] = u
    for axis in range(dim):
        slices1 = tuple(
            slice(None, -1, 2) if i == axis else slice(None)
            for i in range(dim))
        slices2 = tuple(
            slice(1, None, 2) if i == axis else slice(None)
            for i in range(dim))
        uf[slices2] = (uf[slices1] +
                       torch.roll(uf[slices1], shifts=-1, dims=axis)) / 2
    return uf


nx = 129
num_epochs = 100000
epochs = list(range(num_epochs))
lr = 5e-3
L = 1
dx = L / nx
x = torch.linspace(0, L, nx)
uexact = x * (1 - x) / 2

for depth in 1, 3, 5, 7:
    print(f"depth {depth}")
    u = torch.zeros_like(x)
    shape = np.array((np.shape(u)), dtype=int)
    mg = [u.clone()]
    for d in range(depth):
        for s in shape:
            assert (s + 1) % 2 == 0
            assert s > 1
        shape = (shape + 1) // 2
        mg.append(
            torch.zeros(tuple(shape),
                        dtype=u.dtype,
                        requires_grad=u.requires_grad,
                        device=u.device))
    for i in range(len(mg)):
        mg[i].requires_grad = True
    errors = []
    optim = torch.optim.Adam(mg, lr=lr)
    for epoch in epochs:
        optim.zero_grad()
        u = mg[-1]
        for subu in mg[-2::-1]:
            u = upscale_grid(u) + subu
        d2udx2 = (u[:-2] - 2 * u[1:-1] + u[2:]) / (dx**2)
        residuals = d2udx2 + 1
        loss = torch.mean(residuals**2)
        loss += u[0]**2
        loss += u[-1]**2
        loss.backward()
        optim.step()
        e = torch.mean((u - uexact)**2).item()
        errors.append(e)
        if epoch % 10000 == 0:
            print(f"epoch {epoch:05d} error {e:.4e}")
    plt.plot(epochs, errors, label=f"depth = {depth}")
plt.set_xlabel('epoch')
plt.set_ylabel('error')
plt.set_xscale('log')
plt.set_yscale('log')
plt.set_xlim(1, None)
plt.legend()
plt.tight_layout()
plt.show()
