#!/bin/sh

set -eu

mkdir -p results
./forward.py
mpirun -n 8 ./run_plane.py --n 9
./collect_losses.py out_plane/* --out-csv results/losses.csv

propagate() {
    sigma=$1; shift
    sigma_MCMC=$1; shift

    ./generate_posterior_samples.py \
        results/losses.csv --nsamples 128 \
        --out-csv results/samples_sigma_${sigma}.csv \
        --sigma $sigma --sigma-MCMC $sigma_MCMC

    mpirun -n 8 ./run_samples.py \
           results/samples_sigma_${sigma}.csv \
           --base-out-dir out_samples_sigma_${sigma}

    ./extract_uq_levelsets.py \
        out_samples_sigma_$sigma/* \
        --ground-truth out_forward/u_final.npy \
        --out-contours results/contours_sigma_${sigma}.pkl
}

propagate 0.01 0.001
propagate 0.05 0.007
propagate 0.10 0.015
