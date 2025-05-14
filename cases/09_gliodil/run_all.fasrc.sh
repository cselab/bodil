#!/bin/sh

#set -eu

: ${nprocs=32}
: ${tasks_per_node=4}
: ${nsamples=512}
: ${lambda_pde=1000}
: ${lambda_ic=200}

run_case() {
    patient_code=$1; shift
    sigma_data=$1; shift

    outdir=$SCRATCH/uq-odil/gliodil/results_patient_${patient_code}_sigma_data_${sigma_data}_lambda_pde_${lambda_pde}_lambda_ic_${lambda_ic}_nsamples_${nsamples}
    mkdir -p $outdir

    data_path=$SCRATCH/uq-odil/data_GliODIL_essential/data_${patient_code}

    #extra="--restart-from $outdir/out_TMCMC/__checkpoint"
    extra=""

    nodes=$(($nprocs/$tasks_per_node))

    batch=$outdir/sbatch.sh
    cat > $batch <<EOS
#!/bin/bash
#SBATCH --partition=seas_gpu
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=${tasks_per_node}
#SBATCH -t 2-00:00 # time (D-HH:MM)
#SBATCH --job-name=p${patient_code}_s${sigma_data}
#SBATCH --constraint=h100
#SBATCH --gres=gpu:${tasks_per_node}
#SBATCH --mem=40G

module load gcc/14.2.0-fasrc01 openmpi/5.0.5-fasrc01 python/3.12.8-fasrc01
mamba activate uqodil

srun --mpi=pmix -n $nprocs ./run_TMCMC.py \
       $data_path \
       --nsamples $nsamples \
       --base-out-dir $outdir/out_TMCMC \
       --NtNxNyNz 257 64 64 64 \
       --sigma-data $sigma_data \
       --lambda-pde $lambda_pde \
       --lambda-ic $lambda_ic \
       $extra
EOS

    sbatch $batch
}

sigma_data=0.05
run_case 034 $sigma_data
#run_case 020 $sigma_data
