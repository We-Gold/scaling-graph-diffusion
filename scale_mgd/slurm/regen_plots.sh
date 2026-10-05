#!/bin/bash
# Regenerate report Figs 8-9 from a finished run. Submit from scale_mgd/:
#   sbatch --export=ALL,RUN_DIR=outputs/ce_query_edges/ego_2026-10-06_10-00-00 slurm/regen_plots.sh
# Or from an old MQP checkpoint:
#   sbatch --export=ALL,CONFIG=ce_query_edges,DATASET=ego,CHECKPOINT=/path/model_final.pt,OUT_DIR=figs/ego slurm/regen_plots.sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH -J scale-mgd-regen
#SBATCH -p short
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j.out
# GPU type is not recorded. To request one, use e.g. --gres=gpu:a100:1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

if [[ -n "${RUN_DIR:-}" ]]; then
    uv run python scripts/regen_plots.py --run-dir "${RUN_DIR}"
else
    uv run python scripts/regen_plots.py --config "configs/${CONFIG:-ce_query_edges}.yaml" \
        --dataset "${DATASET:?}" --checkpoint "${CHECKPOINT:?}" --out-dir "${OUT_DIR:?}"
fi
