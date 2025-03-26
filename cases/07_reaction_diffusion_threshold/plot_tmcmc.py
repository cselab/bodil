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
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.basedir, "stage_???.json")))


    all_loglike = []
    all_x0 = []
    all_y0 = []

    for path in paths:
        with open(path, "r") as f:
            data = json.load(f)

        stage = data['stage']
        samples = np.array(data['samples'])
        loglike = np.array(data['log_likelihood'])

        idx = np.argsort(loglike)

        all_x0.append(samples[idx,0])
        all_y0.append(samples[idx,1])
        all_loglike.append(loglike[idx])

    all_loglike = np.array(all_loglike)
    vmin = np.quantile(all_loglike, q=0.05)
    vmax = np.quantile(all_loglike, q=0.95)

    for stage in range(len(paths)):
        x0 = all_x0[stage]
        y0 = all_y0[stage]
        ll = all_loglike[stage]

        fig, ax = plt.subplots()
        ax.scatter(x0, y0, c=ll, vmin=vmin, vmax=vmax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$y_0$")
        plt.tight_layout()
        plt.savefig(f"stage-{stage:03d}.png")
        plt.close(fig)


if __name__ == '__main__':
    main()
