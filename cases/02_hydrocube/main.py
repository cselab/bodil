#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_traj', type=str, help="path of beads trajectories, in csv format.")
    args = parser.parse_args()

    csv_path = args.csv_traj

    df = pd.read_csv(csv_path)

    t = df['t'].to_numpy()
    nbeads = len([name for name in df.columns if 'bead' in name]) // 3
    nmotors = len([name for name in df.columns if 'omega' in name])
    nt = len(t)

    beads = np.empty((nbeads, nt, 3))
    omegas = np.empty((nt, nmotors))

    for j in range(nbeads):
        for dim, code in enumerate(['x', 'y', 'z']):
            beads[j,:,dim] = df[f"bead{j}{code}"].to_numpy()

    for k in range(nmotors):
        omegas[:,k] = df[f"omega{k}"].to_numpy()



    fig, axes = plt.subplots(ncols=nbeads, figsize=(nbeads * 4.8,  3.6))

    for j in range(nbeads):
        ax = axes[j]
        for dim, code in enumerate(['x', 'y', 'z']):
            ax.plot(t, beads[j,:,dim], color=f'C{dim}', ls='-')

        ax.set_xlabel(r'$t$')
        ax.set_ylim(0, 1)
        ax.set_ylabel('position')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
