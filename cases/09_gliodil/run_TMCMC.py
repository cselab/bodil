#!/usr/bin/env python

import argparse
import glob
import json
from mpi4py import MPI
import numpy as np
import os
import pandas as pd
import sys
import torch

from gliodil import run_gliodil
from uq_odil.TMCMC import TMCMC
from prepare_data import load_data

def compute_loss(path):
    df = pd.read_csv(os.path.join(path, 'train_history.csv'))
    loss = df['loss'].to_numpy()
    n = len(loss) // 10
    return np.mean(loss[-n:])

def compute_bounds(data_path):
    meta_data, raw_data, trimmed_data = load_data(data_path, trim_scale=1)
    lo = np.array(meta_data['crop_offset'], dtype=float)
    L = np.array(meta_data['crop_extent'], dtype=float)
    hi = lo + L
    return lo, hi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="path to directory containing .nii files")
    parser.add_argument("--base-out-dir", type=str, default="out_TMCMC", help="base output directory")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of samples.")
    parser.add_argument("--seed", type=int, default=21387875, help="Random seed.")
    parser.add_argument("--sigma-data", type=float, default=0.05, help="Per-voxel data segmentation error parameter.")
    parser.add_argument('--NtNxNyNz', type=int, nargs=4, default=[129, 64, 64, 64], help='odil grid size (Nt, Nx, Ny, Nz)')
    parser.add_argument('--restart-from', type=str, default=None, help='if set, restart from this directory')
    parser.add_argument('--lambda-pde', type=float, default=100, help='weight for PDE loss')
    parser.add_argument('--lambda-ic', type=float, default=200, help='weight for IC loss')
    args = parser.parse_args()

    data_path = args.data_path
    base_dir = args.base_out_dir
    nsamples = args.nsamples
    seed = args.seed
    Nt, Nx, Ny, Nz = args.NtNxNyNz
    path_to_restart = args.restart_from
    trim_scale = 1.5
    sigma_data = args.sigma_data

    lambda_ic = args.lambda_ic
    lambda_pde = args.lambda_pde

    lo, hi = compute_bounds(data_path)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cuda_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    beta = Nx * Ny * Ny

    def log_likelihood(sample, context):
        x0, y0, z0 = sample
        stage = context['stage']
        out_dir = os.path.join(base_dir,
                               f"stage_{stage:03d}",
                               f"x0_{x0:.5f}_y0_{y0:.5f}_z0_{z0:.5f}")

        try:
            # test if this was already computed
            vtk_files = glob.glob(os.path.join(out_dir, '*.vtk'))
            if len(vtk_files) == 0:
                run_gliodil(data_path=data_path,
                            Nt=Nt, Nx=Nx, Ny=Ny, Nz=Nz,
                            out_dir=out_dir,
                            xyz0=[x0, y0, z0],
                            dump_raw_to_vtk=False,
                            device=device, num_epochs=5000, lr=1e-3,
                            lambda_pde=lambda_pde, lambda_ic=lambda_ic,
                            sigma_data=sigma_data,
                            verbose=False, trim_scale=trim_scale)
        except FileNotFoundError as e:
            print(f"Rank {rank}: Failed on {MPI.Get_processor_name()}, Device: {device}, Exception: {e}", file=sys.stderr)
            sys.stderr.flush()
            exit(1)
        except Exception as e:
            print(f"Rank {rank}: Failed on {MPI.Get_processor_name()}, Device: {device}, Exception: {e}", file=sys.stderr)
            sys.stderr.flush()
        try:
            loss = compute_loss(path=out_dir)
            return -beta * loss
        except:
            #return -np.inf
            return -1e9

    def prior_sampler(rng):
        # coordinates in simulations are in [0, hi-lo]
        return rng.uniform(lo, hi) - lo

    def log_prior_density(x):
        return - 2 * np.log(np.prod(hi-lo))

    def callback(stage, samples, log_fvals, zeta, S):
        data = {
            'stage': stage,
            'zeta': zeta,
            'S': S,
            'samples': samples.tolist(),
            'log_likelihood': log_fvals.tolist()
        }
        path = os.path.join(base_dir, f"stage_{stage:03d}.json")
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    samples, evidence = TMCMC(log_likelihood=log_likelihood,
                              log_prior_density=log_prior_density,
                              prior_sampler=prior_sampler,
                              beta=0.2,
                              gamma=1,
                              num_samples=nsamples,
                              comm=comm,
                              seed=seed,
                              callback=callback,
                              checkpoint_dir=os.path.join(base_dir, '__checkpoint'),
                              restart_from_dir=path_to_restart)

    if rank == 0:
        print(f"evidence: {evidence}")



if __name__ == '__main__':
    main()
