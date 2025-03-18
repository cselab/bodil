#!/bin/sh

set -eu

for smoothness in 0.125 1.000; do
    for sigma_data in 0.05 0.1; do
        outdir=results_smoothness_${smoothness}_sigma_${sigma_data}
        mkdir -p $outdir
        ./forward.py \
            --smoothness $smoothness \
            --sigma-data $sigma_data \
            --out-dir $outdir/out_forward

        mpirun -n 8 ./run_plane.py \
               --n 19 \
               --forward-dir $outdir/out_forward \
               --base-out-dir $outdir/out_plane \
               --sigma-data $sigma_data

        ./collect_losses.py \
            $outdir/out_plane/* \
            --out-csv $outdir/losses.csv

        ./generate_posterior_samples.py \
            $outdir/losses.csv --nsamples 128 \
            --out-csv $outdir/samples.csv

        mpirun -n 8 ./run_samples.py \
               --forward-dir $outdir/out_forward \
               $outdir/samples.csv \
               --sigma-data $sigma_data \
               --base-out-dir $outdir/out_samples

        ./extract_uq_levelsets.py \
            $outdir/out_samples/* \
            --ground-truth $outdir/out_forward/u_final.npy \
            --out-contours $outdir/contours.pkl
    done
done
