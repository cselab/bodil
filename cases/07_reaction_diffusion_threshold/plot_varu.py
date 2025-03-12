#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_paths', type=str, nargs='+', help='paths of output directories generated from samples')
    parser.add_argument('--diff-field', type=str, default=None, help='paths of diffusion coeff field npy file')
    args = parser.parse_args()

    L = 1
    paths = args.sample_paths

    all_u = []

    for path in paths:
        try:
            with open(os.path.join(path, "u_final.npy"), "rb") as f:
                u = np.load(f)
                all_u.append(u.copy())
        except:
            pass

    all_u = np.array(all_u)
    nsamples, ny, nx = all_u.shape

    x = np.linspace(0, L, nx)
    y = np.linspace(0, L, ny)
    X, Y = np.meshgrid(x, y)

    stdu = np.std(all_u, axis=0)

    fig, ax = plt.subplots()
    ax.contourf(X, Y, stdu, levels=100)

    if args.diff_field:
        with open(args.diff_field, "rb") as f:
            D = np.load(f)
        threshold = (np.min(D) + np.max(D)) / 2
        ax.contour(X, Y, D, levels=[threshold], colors='w')

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')


    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
