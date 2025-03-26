#!/usr/bin/env python

import argparse
from mpi4py import MPI
import numpy as np
import os
import torch

from inverse import run
from uq_odil.TMCMC import TMCMC
import collect_losses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-dir", type=str, default="out_forward", help="output directory of forward.py")
    parser.add_argument("--base-out-dir", type=str, default="out_TMCMC", help="base output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Measurement threshold.")
    parser.add_argument("--sigma-data", type=float, default=0.001, help="Data uncertainty parameter.")
    parser.add_argument("--lambda-pde", type=float, default=10, help="Coefficient for PDE residuals loss.")
    parser.add_argument("--lambda-ic", type=float, default=100, help="Coefficient for IC residuals loss.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of samples.")
    parser.add_argument("--seed", type=int, default=21387875, help="Random seed.")
    args = parser.parse_args()

    forward_dir = args.forward_dir
    base_dir = args.base_out_dir
    threshold = args.threshold
    sigma_data = args.sigma_data
    nsamples = args.nsamples
    seed = args.seed

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    cuda_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(forward_dir, "ut_final.npy"), "rb") as f:
        ut_final = np.load(f)
        ny, nx = ut_final.shape
        del ut_final

    L = 1.0
    beta = nx * ny

    def log_likelihood(sample, context):
        x0, y0 = sample
        stage = context['stage']
        out_dir = os.path.join(base_dir,
                               f"stage_{stage:03d}",
                               f"x0_{x0:.5f}_y0_{y0:.5f}")

        try:
            run(forward_dir=forward_dir,
                out_dir=out_dir,
                initial_pos=[x0, y0],
                dump_snapshots=False,
                threshold=threshold,
                sigma_data=sigma_data,
                lambda_pde=args.lambda_pde,
                lambda_ic=args.lambda_ic,
                device=device,
                verbose=False)
        except RuntimeError as e:
            print(rank, e)
            print(f"Failed on {MPI.Get_processor_name()}, Device: {device}")

        try:
            loss = collect_losses.compute_loss(out_dir)
            return -beta * loss
        except:
            return -np.inf

    def prior_sampler(rng):
        return rng.uniform(0, L, size=2)

    def log_prior_density(x):
        return - 2 * np.log(L)


    samples, evidence = TMCMC(log_likelihood=log_likelihood,
                              log_prior_density=log_prior_density,
                              prior_sampler=prior_sampler,
                              beta=0.2,
                              gamma=1,
                              num_samples=nsamples,
                              comm=comm,
                              seed=seed)

    if rank == 0:
        print(f"evidence: {evidence}")



if __name__ == '__main__':
    main()
