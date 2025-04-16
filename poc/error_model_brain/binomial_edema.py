#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def main():
    u = np.linspace(0, 1, 1000)
    uc1, uc2 = 0.4, 0.7 # thresholds

    # loss computed from the binomial distribution with a sigmoid probability:

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))

    for sigma in [0.05, 0.02, 0.01]:
        # probability of edema given that y_ij = 1
        alpha = sigmoid((u - uc1) / sigma) - sigmoid((u - uc2) / sigma)
        p = alpha
        loss = -np.log(alpha)
        axes[0,0].plot(u, loss, label=fr'$\sigma = {sigma}$')

        p = np.exp(- loss)
        norm = np.sum((p[:-1] + p[1:]) * np.diff(u))
        p /= norm
        axes[1,0].plot(u, p, label=fr'$\sigma = {sigma}$')

        # probability of edema given that y_ij = 0
        alpha = sigmoid((uc1 - u) / sigma) + sigmoid((u - uc2) / sigma)
        p = alpha
        loss = -np.log(alpha)
        axes[0,1].plot(u, loss, label=fr'$\sigma = {sigma}$')

        p = np.exp(- loss)
        norm = np.sum((p[:-1] + p[1:]) * np.diff(u))
        p /= norm
        axes[1,1].plot(u, p, label=fr'$\sigma = {sigma}$')

    ax = axes[0,0]
    ax.axvline(uc1, ls='--', c='k')
    ax.axvline(uc2, ls='--', c='k')
    ax.set_xlabel(r"$u_{ij}$")
    ax.set_ylabel(r"loss for $y_{ij} = 1$")
    ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
    ax.legend()

    ax = axes[0,1]
    ax.axvline(uc1, ls='--', c='k')
    ax.axvline(uc2, ls='--', c='k')
    ax.set_xlabel(r"$u_{ij}$")
    ax.set_ylabel(r"loss for $y_{ij} = 0$")
    ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
    ax.legend()

    ax = axes[1,0]
    ax.axvline(uc1, ls='--', c='k')
    ax.axvline(uc2, ls='--', c='k')
    ax.set_xlabel(r"$u_{ij}$")
    ax.set_ylabel(r"$\alpha_{ij}$ for $y_{ij}=1$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)

    ax = axes[1,1]
    ax.axvline(uc1, ls='--', c='k')
    ax.axvline(uc2, ls='--', c='k')
    ax.set_xlabel(r"$u_{ij}$")
    ax.set_ylabel(r"$\alpha_{ij}$ for $y_{ij}=0$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)

    plt.tight_layout()
    #plt.savefig("error_model.pdf")
    plt.show()

if __name__ == '__main__':
    main()
