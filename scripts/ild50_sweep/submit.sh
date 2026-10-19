#!/bin/bash
# Submit an ILD50 sweep: prefetch array (one job per seed), then the run array
# (one job per chunk of tasks), which starts only after every prefetch succeeded.
#
#   scripts/ild50_sweep/submit.sh <sweep> [chunk_size] [extra driver args...]
#   e.g. scripts/ild50_sweep/submit.sh oat 2   (chunk_size x task time must fit run.slurm --time)
#        scripts/ild50_sweep/submit.sh oat 2 --freq 12000
#
# Env: MAX_PARALLEL (default 50) caps concurrently running run-array jobs.
# Resubmitting the same command only re-runs tasks whose output is missing.
set -euo pipefail
SWEEP=${1:?usage: submit.sh <sweep> [chunk_size] [extra driver args...]}
CHUNK_SIZE=${2:-1}
shift $(( $# < 2 ? $# : 2 ))
EXTRA="$*"

REPO=$(cd "$(dirname "$0")/../.." && pwd)
HERE="$REPO/scripts/ild50_sweep"
LOGS="$REPO/logs/ild50_sweep/$SWEEP"
mkdir -p "$LOGS"

source "$HERE/env.sh"
INFO=$(cd "$REPO/simulate" && python main_ild50_sweep.py list --sweep "$SWEEP" --chunk-size "$CHUNK_SIZE" $EXTRA)
echo "$INFO" | tail -n 4
N_SEEDS=$(echo "$INFO" | sed -n 's/^N_SEEDS=//p')
N_CHUNKS=$(echo "$INFO" | sed -n 's/^N_CHUNKS=//p')

export REPO SWEEP CHUNK_SIZE EXTRA
PF=$(sbatch --parsable --array=0-$((N_SEEDS - 1)) \
    --output="$LOGS/prefetch_%A_%a.out" --export=ALL "$HERE/prefetch.slurm")
RUN=$(sbatch --parsable --array=0-$((N_CHUNKS - 1))%${MAX_PARALLEL:-50} \
    --dependency=afterok:"$PF" \
    --output="$LOGS/run_%A_%a.out" --export=ALL "$HERE/run.slurm")
echo "prefetch job: $PF  ($N_SEEDS seeds)"
echo "run job:      $RUN ($N_CHUNKS chunks of $CHUNK_SIZE)"
