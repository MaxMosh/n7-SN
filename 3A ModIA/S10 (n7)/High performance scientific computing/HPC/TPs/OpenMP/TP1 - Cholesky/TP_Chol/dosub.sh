#!/usr/bin/bash

#Job name
#SBATCH -J mmoshfeg
# Asking for one node
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p small
#SBATCH --ntasks-per-node=1
# Output results message
#SBATCH -o chol%j.out
# Output error message
#SBATCH -e chol%j.err
#SBATCH -t 0:10:00
##SBATCH --exclusive

module purge
source ${HOME}/mmoshfeg/TP_Chol/env_cpuonly.sh
export OMP_MAX_TASK_PRIORITY=999

cd ${SLURM_SUBMIT_DIR}

./bench_strong 200 60

./bench_weak 200 20
