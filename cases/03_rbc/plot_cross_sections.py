#!/usr/bin/env python3

import argparse
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import trimesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('samples_dir', type=str, help='directory that contains mesh samples')
    parser.add_argument('--out', type=str, default=None, help='output plot')
    args = parser.parse_args()

    mesh_paths = sorted(glob.glob(os.path.join(args.samples_dir, "*.ply")))
    out = args.out

    fig, ax = plt.subplots()

    for mesh_path in mesh_paths:
        mesh = trimesh.load(mesh_path)
        lines = trimesh.intersections.mesh_plane(mesh,
                                                 plane_normal=(0, 0, 1),
                                                 plane_origin=(0, 0, 0))
        lines = np.array(lines)
        x = lines[:,:,0]
        y = lines[:,:,1]
        ax.plot(x.T, y.T, '-', color='r', alpha=1, lw=0.001)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r"$x$ ($\mu$m)")
    ax.set_ylabel(r"$z$ ($\mu$m)")
    plt.tight_layout()

    if out is None:
        plt.show()
    else:
        plt.savefig(out)


if __name__ == '__main__':
    main()
