#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def main():
    uc = 0.7
    sigma = 0.03
    u = np.linspace(0, 1, 1000)
    fdouble = 0.5 + 0.5 * np.sign(u - uc) * (1 - np.exp(-(u-uc)**2/sigma**2))
    f = 0.5 * (1 + np.tanh((u - uc) / sigma))

    fig, ax = plt.subplots()
    ax.plot(u, f, label='sigmoid')
    ax.plot(u, fdouble, label='Benze et al.')
    ax.set_xlabel(r"$u$")
    ax.set_ylabel(r"$\alpha(u, u_c)$")
    ax.set_xlim(0, 1)
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
