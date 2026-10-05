#!/bin/bash
# Train + evaluate Scale-MGD on one dataset. Submit from scale_mgd/:
#   sbatch --export=ALL,CONFIG=ce_query_edges,DATASET=planar slurm/run.sh
# CONFIG: ce_query_edges (default, report runs) | vlb_query_edges
# DATASET: zinc250k | planar | ego
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH -J scale-mgd
#SBATCH -p short
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out
# GPU type is not recorded for the original runs. To request one, use e.g.
#   #SBATCH --gres=gpu:a100:1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

CONFIG="${CONFIG:-ce_query_edges}"
DATASET="${DATASET:?set DATASET=zinc250k|planar|ego}"

uv run python scripts/run.py --config "configs/${CONFIG}.yaml" --dataset "${DATASET}"
