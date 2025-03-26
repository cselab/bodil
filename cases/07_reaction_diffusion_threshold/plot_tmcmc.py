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

    for j, path in enumerate(paths):
        print(path)

        with open(path, "r") as f:
            data = json.load(f)

        stage = data['stage']
        samples = np.array(data['samples'])
        loglike = np.array(data['log_likelihood'])

        x0 = samples[:,0]
        y0 = samples[:,1]

        fig, ax = plt.subplots()
        ax.scatter(x0, y0, c=loglike)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$y_0$")
        plt.tight_layout()
        plt.savefig(f"stage-{stage:03d}.png")


if __name__ == '__main__':
    main()
