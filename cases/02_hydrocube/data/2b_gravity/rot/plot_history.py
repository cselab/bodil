#!/usr/bin/env python3

from odil import plotutil
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from matplotlib import colors
import argparse

def rotate_xyz(p, rot, reverse=False):
    '''
    p: array, shape (..., 3)
    '''
    p = np.copy(p)
    if reverse:
        rot = (4 - rot) % 4
    if rot == 0:
        pass
    elif rot == 1:
        p[..., 1] = 1 - p[..., 1]
        p[..., (0, 1)] = p[..., (1, 0)]
    elif rot == 2:
        p[..., 0] = 1 - p[..., 0]
        p[..., 1] = 1 - p[..., 1]
    elif rot == 3:
        p[..., 0] = 1 - p[..., 0]
        p[..., (0, 1)] = p[..., (1, 0)]
    else:
        raise RuntimeError('Unknown rot={}'.format(rot))
    return p

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

# Shape (Nbeads=2, Nt, 3).
ttargets = np.array([[rotate_xyz(target, rot) for rot in u['rot']]
                     for target in targets])

fig, axes = plt.subplots(3, 1, figsize=(5, 7))
t = u['t']
for i, (bead, ax) in enumerate(zip(['bead0', 'bead1'], axes[:2])):
    ax.plot(t, u[bead + 'x'], c='C0', label='x')
    ax.plot(t, u[bead + 'y'], c='C1', label='y')
    ax.plot(t, u[bead + 'z'], c='C2', label='z')

    ax.plot(t, ttargets[i, :, 0], c='C0', ls='--', zorder=0, lw=0.5, label='x target')
    ax.plot(t, ttargets[i, :, 1], c='C1', ls='--', zorder=0, lw=0.5, label='y target')
    ax.plot(t, ttargets[i, :, 2], c='C2', ls='--', zorder=0, lw=0.5, label='z target')

    ax.set_ylabel('position [-]')
    ax.set_ylim(0, 1)
    if i == 0:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title('Bead {:}'.format(i))

for ax in axes:
    ax.set_xlim(0, 200)

ax = axes[2]
ax.set_title('Distance to target')
beads = np.swapaxes([[u['bead0x'], u['bead0y'], u['bead0z']],
                     [u['bead1x'], u['bead1y'], u['bead1z']]], 2, 1)
dist = np.sum(np.linalg.norm(beads - ttargets, axis=2), axis=0)
for it in range(1, len(t)):
    if any(ttargets[0, it] != ttargets[0, it - 1]):
        ax.axvline(t[it], c='k', lw=0.5, ls='--')
ax.set_xlabel('time [s]')
ax.set_ylabel('distance [-]')
ax.plot(t, dist, c='C3')
fig.tight_layout()
plotutil.savefig(fig, 'beads_dist')
