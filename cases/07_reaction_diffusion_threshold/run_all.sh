#!/bin/sh

mkdir -p results
./forward.py
mpirun -n 8 ./run_plane.py --n 9
./collect_losses.py out_plane/* --out-csv results/losses.csv
./generate_posterior_samples.py results/losses.csv --nsamples 128 --out-csv results/samples.csv
mpirun -n 8 ./run_samples.py results/samples.csv
