#!/usr/bin/env python3

from odil import plotutil
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from matplotlib import colors
import argparse

plotutil.set_extlist(['pdf'])

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('csvpath',
                    type=str,
                    default='*.csv',
                    nargs='?',
                    help='Path or glob pattern to history_*.csv')
args = parser.parse_args()

csvpath = glob(args.csvpath)[0]

u = np.genfromtxt(csvpath, names=True, delimiter=',')

targets = [[0.3, 0.3, 0.5], [0.3, 0.7, 0.5]]

fig, axes = plt.subplots(2, 1, figsize=(3, 3))
t = u['t']
for i, (bead, ax) in enumerate(zip(['bead0', 'bead1'], axes)):
    ax.plot(t, u[bead + 'x'], c='C0', label='x')
    ax.plot(t, u[bead + 'y'], c='C1', label='y')
    ax.plot(t, u[bead + 'z'], c='C2', label='z')

    policy = np.where(u['policy'] == -1, 1, 0)
    pc = ['C3', 'C4']
    cmap = colors.ListedColormap(pc)
    ax.imshow(np.tile(policy[None, :], (2, 1)),
              cmap=cmap,
              extent=(0, t.max(), 0, 1),
              interpolation='none',
              aspect='auto',
              vmin=0,
              vmax=len(pc),
              alpha=0.2)
    ax.scatter([], [], label='point', c='C3', marker='s', s=10)
    ax.scatter([], [], label='none', c='C4', marker='s', s=10)

    ax.set_xlim(0, 60)

    ones = np.ones_like(t)
    ax.plot(t, ones * targets[i][0], c='C0', ls='--', zorder=0, lw=0.5)
    ax.plot(t, ones * targets[i][1], c='C1', ls='--', zorder=0, lw=0.5)
    ax.plot(t, ones * targets[i][2], c='C2', ls='--', zorder=0, lw=0.5)

    if i == 1:
        ax.set_xlabel('time [s]')
    ax.set_ylabel('position [-]')
    ax.set_ylim(0, 1)
    if i == 0:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title('Bead {:}'.format(i))
fig.tight_layout()
plotutil.savefig(fig, 'beads')
