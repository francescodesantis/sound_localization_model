#!/bin/bash
#SBATCH --job-name=ild50_prefetch
#SBATCH --account=EUHPC_D36_180
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
# One array element per seed: runs the cochlea (no NEST) and fills the ANF cache.
# Submitted by submit.sh, which sets REPO, SWEEP and EXTRA.
set -euo pipefail
source "$REPO/scripts/ild50_sweep/env.sh"
cd "$REPO/simulate"
python main_ild50_sweep.py prefetch --sweep "$SWEEP" --seed "$SLURM_ARRAY_TASK_ID" $EXTRA
