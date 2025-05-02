#!/bin/sh

#set -eu

: ${nprocs=8}
: ${tasks_per_node=1}
: ${nsamples=128}

run_case() {
    patient_code=$1; shift

    outdir=$SCRATCH/uq-odil/gliodil/results_patient_${patient_code}
    mkdir -p $outdir

    data_path=$SCRATCH/uq-odil/data_GliODIL_essential/data_${patient_code}

    nodes=$(($nprocs/$tasks_per_node))

    batch=$outdir/sbatch.sh
    cat > $batch <<EOS
#!/bin/bash
#SBATCH --partition=seas_gpu
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=${tasks_per_node}
#SBATCH -t 2-00:00 # time (D-HH:MM)
#SBATCH --job-name=p${patient_code}
#SBATCH --constraint=h100
#SBATCH --gres=gpu
#SBATCH --mem=40G

module load gcc/14.2.0-fasrc01 openmpi/5.0.5-fasrc01 python/3.12.8-fasrc01
mamba activate uqodil

srun --mpi=pmix -n $nprocs ./run_TMCMC.py \
       $data_path \
       --nsamples $nsamples \
       --base-out-dir $outdir/out_TMCMC \
       --NtNxNyNz 257 64 64 64
EOS

    sbatch $batch
}

run_case 034
