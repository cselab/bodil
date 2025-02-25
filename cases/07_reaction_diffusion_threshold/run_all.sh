#!/bin/sh

./forward.py
mpirun -n 8 ./run_plane.py
./collect_losses.py out_plane/* --out-csv results/losses.csv
./generate_posterior_samples.py results/losses.csv --nsamples 128 --out-csv results/samples.csv
mpirun -n 8 ./run_samples.py results/samples.csv
