#!/usr/bin/env python

import argparse
from mpi4py import MPI
import numpy as np
import os
import pandas as pd
import torch

from inverse import run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("samples_csv", type=str, help="csv files that contain samples of (x0, y0)")
    parser.add_argument("--forward-dir", type=str, default="out_forward", help="output directory of forward.py")
    parser.add_argument("--base-out-dir", type=str, default="out_samples", help="base output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Measurement threshold.")
    parser.add_argument("--sigma-data", type=float, default=0.001, help="Data uncertainty parameter.")
    parser.add_argument("--lambda-pde", type=float, default=10, help="Coefficient for PDE residuals loss.")
    parser.add_argument("--lambda-ic", type=float, default=100, help="Coefficient for IC residuals loss.")
    args = parser.parse_args()

    samples_csv = args.samples_csv
    forward_dir = args.forward_dir
    base_dir = args.base_out_dir
    threshold = args.threshold
    sigma_data = args.sigma_data

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cuda_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(samples_csv)
    X = df['x0'].to_numpy()
    Y = df['y0'].to_numpy()

    n = len(X)
    work_per_rank = (n + size - 1) // size

    start = rank * work_per_rank
    end = min([start + work_per_rank, n])

    for i in range(start, end):
        x0 = X[i]
        y0 = Y[i]
        out_dir = os.path.join(base_dir, f"sample_{i:06d}_x0_{x0:.5f}_y0_{y0:.5f}")
        try:
            run(forward_dir=forward_dir,
                out_dir=out_dir,
                initial_pos=[x0, y0],
                dump_snapshots=False,
                threshold=threshold,
                sigma_data=sigma_data,
                lambda_pde=args.lambda_pde,
                lambda_ic=args.lambda_ic,
                device=device)
        except RuntimeError as e:
            print(e)
            print(f"Failed on {MPI.Get_processor_name()}")
            print(f"Device: {device}")


if __name__ == '__main__':
    main()
