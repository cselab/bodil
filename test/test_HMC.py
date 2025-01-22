#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch

from uq_odil.HMC import HMC

def main():
    device = 'cpu'
    dim = 1
    num_samples = 2000

    x = torch.zeros(dim, requires_grad=True)
    sigma = torch.FloatTensor([1.0] * dim)

    hmc = HMC([x], dt=0.05, L=5, M=0.2)

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

    fig, ax = plt.subplots()
    ax.hist(samples[:,0], range=(-2, 2), bins=50)
    plt.show()

if __name__ == '__main__':
    main()
