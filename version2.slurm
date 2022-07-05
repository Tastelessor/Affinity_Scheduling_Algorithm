#!/bin/sh
#SBATCH --account=dc134-s2168291
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --time=30:00

module load intel-compilers-19
cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=1
srun --cpu-bind=cores ./v2
export OMP_NUM_THREADS=2
srun --cpu-bind=cores ./v2
export OMP_NUM_THREADS=4
srun --cpu-bind=cores ./v2
export OMP_NUM_THREADS=6
srun --cpu-bind=cores ./v2
export OMP_NUM_THREADS=8
srun --cpu-bind=cores ./v2
export OMP_NUM_THREADS=12
srun --cpu-bind=cores ./v2
export OMP_NUM_THREADS=16
srun --cpu-bind=cores ./v2
export OMP_NUM_THREADS=24
srun --cpu-bind=cores ./v2
export OMP_NUM_THREADS=32
srun --cpu-bind=cores ./v2

