#!/bin/sh

set -eu

for smoothness in 0.125 1.000; do
    for sigma_data in 0.0005 0.001 0.005; do
        outdir=results_smoothness_${smoothness}_sigma_${sigma_data}
        mkdir -p $outdir
        ./forward.py \
            --smoothness $smoothness \
            --sigma-data $sigma_data \
            --out-dir $outdir/out_forward

        mpirun -n 8 ./run_plane.py \
               --n 9 \
               --forward-dir $outdir/out_forward \
               --base-out-dir $outdir/out_plane \
               --sigma-data $sigma_data

        ./collect_losses.py \
            $outdir/out_plane/* \
            --out-csv $outdir/losses.csv

    done
done
