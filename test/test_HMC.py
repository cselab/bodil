#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch

from uq_odil.HMC import HMC

def main():
    device = 'cpu'
    dim = 1
    num_samples = 10000
    sig = 2.0

    x = torch.zeros(dim, requires_grad=True)
    sigma = torch.FloatTensor([sig] * dim)

    hmc = HMC([x], dt=0.05, L=10, M=1.0)

    def closure():
        hmc.zero_grad()
        U = torch.sum(x**2 / (2 * sigma**2))
        U.backward()
        return U

    samples = []

    for k in range(num_samples):
        x_, H = hmc.step(closure)
        samples.append(x_[0].detach().numpy())

    samples = np.array(samples)

    r = 4 * sig
    x = np.linspace(-r, r, 256)
    p = np.exp(-x**2 / (2 * sig**2)) / np.sqrt(2 * np.pi * sig**2)

    fig, ax = plt.subplots()
    ax.hist(samples[:,0], range=(-r, r), bins=50, density=True)
    ax.plot(x, p, '-r')
    ax.set_xlim(-r, r)
    ax.set_ylim(0, 1.3 * np.max(p))
    plt.show()

if __name__ == '__main__':
    main()
