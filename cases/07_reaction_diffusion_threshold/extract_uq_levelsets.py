#!/usr/bin/env python

import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_paths', type=str, nargs='+', help='paths of output directories generated from samples')
    parser.add_argument('--ground-truth', type=str, default=False, help='true concentration field (output of forward)')
    parser.add_argument('--out-contours', type=str, default="contours.pkl", help='path to file that will contain the contours')
    parser.add_argument('--show-plot', action='store_true', default=False, help='show plot')
    args = parser.parse_args()

    ground_truth_path = args.ground_truth
    levels = [0.1, 0.3, 0.5, 0.6]
    nlevels = len(levels)
    L = 1
    paths = args.sample_paths

    all_segments = [[] for _ in range(nlevels)]

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
                for i in range(nlevels):
                    all_segments[i] += cont.allsegs[i]
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

    # add fake data for legend
    for i, l in enumerate(levels):
        ax.plot([], [], '-', c=f"C{i}", label=fr"$u(x, y) = {l}$")
    if ground_truth_path:
        ax.plot([], [], '--k', label="ground truth")

    ax.legend(frameon=False, handlelength=1, loc='upper left')

    plt.tight_layout()
    if args.show_plot:
        plt.show()
    plt.close()

    data = {"levels": levels,
            "segments": all_segments}

    if ground_truth_path:
        data["gt_segments"] = cont.allsegs

    with open(args.out_contours, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
