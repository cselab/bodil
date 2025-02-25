#!/usr/bin/env python

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_paths', type=str, nargs='+', help='paths of output directories generated from samples')
    parser.add_argument('--ground-truth', type=str, default=False, help='true concentration field (output of forward)')
    args = parser.parse_args()

    ground_truth_path = args.ground_truth
    levels=[0.5, 0.6, 0.7, 0.8]
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
                cont = ax.contour(X, Y, u,
                                  levels=levels,
                                  colors=[f'C{i}' for i in range(len(levels))],
                                  linewidths=0.1,
                                  vmin=0, vmax=1)
        except:
            pass

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    if ground_truth_path:
        with open(ground_truth_path, "rb") as f:
            u = np.load(f)
            ny, nx = u.shape
            x = np.linspace(0, L, nx)
            y = np.linspace(0, L, ny)
            X, Y = np.meshgrid(x, y)
            cont = ax.contour(X, Y, u,
                              levels=levels,
                              colors='k',
                              linewidths=1,
                              linestyles='--')
            print(u.min(), u.max())

    # add fake data for legend
    for i, l in enumerate(levels):
        ax.plot([], [], '-', c=f"C{i}", label=fr"$u(x, y) = {l}$")

    ax.legend()
    #cb = fig.colorbar(cont, ax=ax)

    # make colorbar lines thicker so we can see the colors
    # see https://stackoverflow.com/a/19372610
    # lw = 2
    # for c in cb.ax.get_children():
    #     if isinstance(c, matplotlib.collections.LineCollection):
    #         num_lines = len(c.get_linewidths())
    #         if num_lines == len(levels):
    #             c.set_linewidths([lw] * num_lines)

    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()
