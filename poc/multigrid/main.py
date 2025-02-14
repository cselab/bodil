#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def coarsen_field(u, levels):
    pass

def main():
    nx = 128
    x = np.linspace(0, 2 * np.pi, nx)
    u = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, u)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u$')
    plt.show()

if __name__ == '__main__':
    main()
