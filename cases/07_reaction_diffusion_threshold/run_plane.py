#!/usr/bin/env python

import argparse
from mpi4py import MPI
import numpy as np
import os
import torch

from inverse import run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-dir", type=str, default="out_forward", help="output directory of forward.py")
    parser.add_argument("--base-out-dir", type=str, default="out_plane", help="base output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Measurement threshold.")
    parser.add_argument("--n", type=int, default=19, help="Number of points along each dimension.")
    args = parser.parse_args()

    forward_dir = args.forward_dir
    base_dir = args.base_out_dir
    threshold = args.threshold
    nx = ny = args.n

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cuda_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    x0s = np.linspace(0.05, 0.95, nx)
    y0s = np.linspace(0.05, 0.95, ny)

    Y, X = np.meshgrid(y0s, x0s)
    X = X.flatten()
    Y = Y.flatten()

    n = len(X)
    work_per_rank = (n + size - 1) // size

    start = rank * work_per_rank
    end = min([start + work_per_rank, n])

    for i in range(start, end):
        x0 = X[i]
        y0 = Y[i]
        out_dir = os.path.join(base_dir, f"x0_{x0:.5f}_y0_{y0:.5f}")
        run(forward_dir=forward_dir,
            out_dir=out_dir,
            initial_pos=[x0, y0],
            dump_snapshots=False,
            threshold=threshold,
            device=device)

if __name__ == '__main__':
    main()
