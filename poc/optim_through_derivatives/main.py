#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam

def main():
    k = 1
    dim = 2

    def energy(x):
        return k * torch.sum(x**2)

    x = torch.ones(dim, requires_grad=True)

    # test 1. minimize energy
    optim = Adam([x], lr=1e-2)

    epochs = list(range(1000))
    losses_energy = []
    for epoch in epochs:
        optim.zero_grad()
        loss = energy(x)
        loss.backward()
        optim.step()
        losses_energy.append(loss.item())
        if epoch % 100 == 0:
            print(f"epoch {epoch:04d} loss {loss.item()}")

    print(x.detach().numpy())


    # test 2. compute derivative and minimize loss.

    x = torch.ones(dim, requires_grad=True)

    # test 1. minimize energy
    optim = Adam([x], lr=1e-2)

    losses_forces = []
    for epoch in epochs:
        optim.zero_grad()
        neg_E = -energy(x)
        forces = torch.autograd.grad(neg_E, x, create_graph=True, materialize_grads=True)[0]
        loss = 1/4 * torch.sum(forces**2)
        loss.backward()
        optim.step()
        losses_forces.append(loss.item())
        if epoch % 100 == 0:
            print(f"epoch {epoch:04d} loss {loss.item()}")

    print(x.detach().numpy())

    fig, ax = plt.subplots()
    ax.plot(epochs, losses_energy, '-', label='energy')
    ax.plot(epochs, losses_forces, '--', label='forces')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
