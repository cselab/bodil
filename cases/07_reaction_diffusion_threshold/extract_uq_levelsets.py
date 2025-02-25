#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_paths', type=str, nargs='+', help='paths of output directories generated from samples')
    args = parser.parse_args()

    L = 1
    paths = args.sample_paths

    fig, ax = plt.subplots()

    for path in paths:
        try:
            with open(os.path.join(path, "u_final.npy"), "rb") as f:
                u = np.load(f)
                ny, nx = u.shape
                x = np.linspace(0, L, nx)
                y = np.linspace(0, L, ny)
                X, Y = np.meshgrid(x, y)
                cont = ax.contour(X, Y, u, levels=[0.1, 0.3, 0.5, 0.7], linewidths=0.1)
        except:
            pass

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    fig.colorbar(cont, ax=ax)
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
