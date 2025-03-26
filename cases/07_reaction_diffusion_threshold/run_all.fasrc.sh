#!/bin/sh

#set -eu

: ${nprocs=32}
: ${tasks_per_node=4}
: ${nsamples=256}

module load gcc/14.2.0-fasrc01 openmpi/5.0.5-fasrc01 python/3.12.8-fasrc01
mamba activate uqodil

run_case() {
    smoothness=$1; shift
    sigma_data=$1; shift
    lambda_pde=$1; shift
    lambda_ic=$1; shift

    outdir=$SCRATCH/uq-odil/reaction_diffusion/results_smoothness_${smoothness}_sigma_${sigma_data}_lambda_pde_${lambda_pde}_ic_${lambda_ic}
    mkdir -p $outdir

    nodes=$(python -c "print($nprocs//$tasks_per_node)")

    batch=$outdir/sbatch.sh
    cat > $batch <<EOS
#!/bin/bash
#SBATCH --partition=seas_gpu
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=${tasks_per_node}
#SBATCH -t 0-12:00 # time (D-HH:MM)
#SBATCH --job-name=TMCMC-${smoothness}_${sigma_data}
#SBATCH --constraint=h100
#SBATCH --gres=gpu
#SBATCH --mem=20G

./forward.py \
    --smoothness $smoothness \
    --sigma-data $sigma_data \
    --out-dir $outdir/out_forward

srun --mpi=pmix -n $nprocs ./run_TMCMC.py \
       --nsamples $nsamples \
       --forward-dir $outdir/out_forward \
       --base-out-dir $outdir/out_TMCMC \
       --sigma-data $sigma_data \
       --lambda-pde $lambda_pde \
       --lambda-ic $lambda_ic

for path in \$(find $outdir/out_TMCMC/ -type d -name "stage_???"); do
    code=\$(basename \$path)
    ./extract_uq_levelsets.py \
        $outdir/out_TMCMC/\$code/* \
        --ground-truth $outdir/out_forward/u_final.npy \
        --out-contours $outdir/contours_\$code.pkl
done
EOS

    sbatch $batch
}

run_case 0.125 0.05 10 100
run_case 0.125 0.10 40 400
run_case 0.125 0.01 10 100

run_case 1.000 0.05 10 100
run_case 1.000 0.10 40 400
run_case 1.000 0.01 10 100
