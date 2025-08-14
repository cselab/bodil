#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch

from uq_odil.HMC import HMC

def main():
    device = 'cpu'
    dim = 1
    num_samples = 5000
    sig = 2.0
    burnin = 10

    x = torch.zeros(dim, requires_grad=True)
    sigma = torch.FloatTensor([sig] * dim)

    hmc = HMC([x], dt=2.55 * sig, L=10, M=0.5 * sig**2)

    def closure():
        hmc.zero_grad()
        U = torch.sum(x**2 / (2 * sigma**2))
        U.backward()
        return U

    samples = []
    num_accepted = 0
    for k in range(num_samples + burnin):
        x_, H, U, accepted = hmc.step(closure)
        if k >= burnin:
            num_accepted += accepted
            samples.append(x_[0].detach().numpy())

    print(f"accptence rate: {num_accepted/num_samples}")
    samples = np.array(samples)

    r = 4 * sig
    x = np.linspace(-r, r, 256)
    p = np.exp(-x**2 / (2 * sig**2)) / np.sqrt(2 * np.pi * sig**2)

    fig, ax = plt.subplots()
    ax.hist(samples[:,0], range=(-r, r), bins=50, density=True, label='Hamiltonian MC')
    ax.plot(x, p, '-r', label='exact')
    ax.set_xlim(-r, r)
    ax.set_ylim(0, 1.3 * np.max(p))
    ax.set_xlabel(r'$x$')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
