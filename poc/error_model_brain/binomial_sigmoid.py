#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
    #return 0.5 * (1 + np.tanh(x))

def relu(x):
    return np.maximum(x, 0)

def main():
    u = np.linspace(0, 1, 1000)
    uc = 0.7 # threshold

    # loss computed from the binomial distribution with a sigmoid probability:

    fig, axes = plt.subplots(ncols=2, figsize=(6.8, 3.4))

    for sigma in [0.05, 0.02, 0.01]:
        alpha = sigmoid((u - uc) / sigma)
        p = alpha
        loss = -np.log(alpha)
        axes[0].plot(u, loss, label=fr'$\sigma = {sigma}$')
        relu_approx = relu(-(u-uc)/sigma)
        axes[0].plot(u, relu_approx, '--k')

        p = np.exp(- loss)
        norm = np.sum((p[:-1] + p[1:]) * np.diff(u))
        p /= norm
        axes[1].plot(u, p, label=fr'$\sigma = {sigma}$')

        papprox = np.exp(-relu_approx)
        papprox /= np.sum((papprox[:-1] + papprox[1:]) * np.diff(u))
        axes[1].plot(u, papprox, '--k')

    ax = axes[0]
    ax.axvline(uc, ls='--', c='b')
    ax.set_xlabel(r"$u_{ij}$")
    ax.set_ylabel(r"loss for $y_{ij} = 1$")
    ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
    ax.legend()

    ax = axes[1]
    ax.axvline(uc, ls='--', c='b')
    ax.set_xlabel(r"$u_{ij}$")
    ax.set_ylabel(r"$\alpha_{ij}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)

    plt.tight_layout()
    plt.savefig("error_model.pdf")
    plt.show()

if __name__ == '__main__':
    main()
