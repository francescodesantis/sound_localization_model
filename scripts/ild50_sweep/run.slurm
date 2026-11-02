#!/bin/bash
#SBATCH --job-name=ild50_run
#SBATCH --account=EUHPC_D36_180
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60GB
#SBATCH --time=02:00:00
# One array element per chunk of (config, seed) tasks.
# Keep --cpus-per-task identical for the whole sweep: the NEST random streams
# depend on the number of virtual processes, so changing it breaks the
# common-seed pairing between configurations.
# Submitted by submit.sh, which sets REPO, SWEEP, CHUNK_SIZE and EXTRA.
set -euo pipefail
source "$REPO/scripts/ild50_sweep/env.sh"
cd "$REPO/simulate"
python main_ild50_sweep.py run --sweep "$SWEEP" --chunk "$SLURM_ARRAY_TASK_ID" \
    --chunk-size "$CHUNK_SIZE" --threads "$SLURM_CPUS_PER_TASK" $EXTRA
