#!/usr/bin/env python

import argparse
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('basedir', type=str, help='directory containing json files.')
    parser.add_argument('--variables', nargs=2, default=['x0', 'y0'], help='variables to plot')
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.basedir, "stage_???.json")))
    var0, var1 = args.variables

    all_loglike = []
    all_x0 = []
    all_y0 = []
    all_z0 = []

    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)

        stage = data['stage']
        samples = np.array(data['samples'])
        loglike = np.array(data['log_likelihood'])

        idx = np.argsort(loglike)

        all_x0.append(samples[idx,0])
        all_y0.append(samples[idx,1])
        all_z0.append(samples[idx,2])
        all_loglike.append(loglike[idx])

    all_loglike = np.array(all_loglike)
    vmin = np.quantile(all_loglike, q=0.5)
    vmax = np.quantile(all_loglike, q=1)

    var2vals = {
        'x0': np.array(all_x0),
        'y0': np.array(all_y0),
        'z0': np.array(all_z0)
    }

    var2lo = {}
    var2hi = {}
    for key, vals in var2vals.items():
        var2lo[key] = np.min(vals)
        var2hi[key] = np.max(vals)

    for stage in range(len(paths)):
        x = var2vals[var0][stage]
        y = var2vals[var1][stage]
        ll = all_loglike[stage]

        idx = np.argwhere(ll == -1e-9)
        ll[idx] = np.nan

        fig, ax = plt.subplots()
        ax.scatter(x, y, c=ll, vmin=vmin, vmax=vmax)
        ax.set_xlim(var2lo[var0], var2hi[var0])
        ax.set_ylim(var2lo[var1], var2hi[var1])
        ax.set_aspect('equal')
        ax.set_xlabel(var0)
        ax.set_ylabel(var1)
        plt.tight_layout()
        plt.savefig(f"stage-{stage:03d}.png")
        plt.close(fig)


if __name__ == '__main__':
    main()
