#!/bin/sh

set -eu

for smoothness in 0.125 1.000; do
    outdir=results_smoothness_${smoothness}
    mkdir -p $outdir
    ./forward.py --smoothness $smoothness --out-dir $outdir/out_forward
    mpirun -n 8 ./run_plane.py --n 9 --forward-dir $outdir/out_forward --base-out-dir $outdir/out_plane
    ./collect_losses.py $outdir/out_plane/* --out-csv $outdir/losses.csv

    propagate() {
        sigma=$1; shift
        sigma_MCMC=$1; shift

        ./generate_posterior_samples.py \
            $outdir/losses.csv --nsamples 128 \
            --out-csv $outdir/samples_sigma_${sigma}.csv \
            --sigma $sigma --sigma-MCMC $sigma_MCMC

        mpirun -n 8 ./run_samples.py \
               $outdir/samples_sigma_${sigma}.csv \
               --base-out-dir $outdir/out_samples_sigma_${sigma}

        ./extract_uq_levelsets.py \
            $outdir/out_samples_sigma_$sigma/* \
            --ground-truth $outdir/out_forward/u_final.npy \
            --out-contours $outdir/contours_sigma_${sigma}.pkl
    }

    propagate 0.01 0.001
    propagate 0.05 0.007
    propagate 0.10 0.015
done
