# Sourced by the ILD50 sweep jobs and by submit.sh (CINECA Leonardo setup).
module purge
source /leonardo_work/EUHPC_D36_180/miniforge3/etc/profile.d/conda.sh
conda activate sl_env
ulimit -s unlimited

# NEST threads are set through the kernel (--threads); keep libraries single-threaded
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
