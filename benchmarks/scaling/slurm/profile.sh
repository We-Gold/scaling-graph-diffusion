#!/bin/bash
#SBATCH -J scaling
#SBATCH -p short
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH -t 08:00:00
# Report 5.1: 1x A100-80G. The Turing GPU type name for --gres is not recorded.
# Check with: sinfo -o "%P %G". Then use for example --gres=gpu:a100:1
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x-%j.out

# Profile one model on the 6 graph types of report Table 2 (10 seeds). One model per job, so the
# RAM column is per process, as in the original runs. The job log has the same text format as
# the old logs in results/. The CSV goes to out/raw/<model>.csv.
# Usage, from benchmarks/scaling/:
#   mkdir -p logs && sbatch slurm/profile.sh <scale_mgd|mg_diff|digress|sparsediff|sparserdiff>
set -euo pipefail

MODEL=${1:?first argument: scale_mgd, mg_diff, digress, sparsediff or sparserdiff}
OUT=out/raw/$MODEL.csv
nvidia-smi || true

case "$MODEL" in
  scale_mgd) uv run python -m profile_scale_mgd --out "$OUT" ;;   # 20 steps (as in tables 3-8)
  mg_diff)   uv run python -m profile_mg_diff --out "$OUT" ;;
  digress)   uv run --project ../../sparserdiff python -m profile_digress --out "$OUT" ;;
  sparsediff)
    uv run --project ../../sparserdiff python -m profile_sparsediff --sampler original --out "$OUT" ;;
  sparserdiff)
    uv run --project ../../sparserdiff python -m profile_sparsediff --sampler novel --out "$OUT" ;;
  *) echo "unknown model: $MODEL" >&2; exit 1 ;;
esac
