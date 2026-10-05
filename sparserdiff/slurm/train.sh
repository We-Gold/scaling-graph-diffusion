#!/bin/bash
#SBATCH -J sparserdiff
#SBATCH -p short
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH -t 23:00:00
# Report: 1x NVIDIA L40S. The Turing GPU type name for --gres is not recorded.
# Use --gres=gpu:<type>:1 once it is confirmed.
#SBATCH --gres=gpu:1
#SBATCH -o logs/%x-%j.out

# Train SparseDiff (flag false) or SparserDiff (flag true) for report tables 12-13.
# Usage, from sparserdiff/:  mkdir -p logs && sbatch slurm/train.sh <qm9|ego> <true|false>
set -euo pipefail

DATASET=${1:?first argument: qm9 or ego}
NOVEL=${2:?second argument: true or false}
VARIANT=$([ "$NOVEL" = true ] && echo sparserdiff || echo sparsediff)

# Hydra entry point must run from sparse_diffusion/ (data and outputs resolve to ../).
cd sparse_diffusion
if [ "$DATASET" = qm9 ]; then
  # Table 12: QM9 without H, 20 epochs, batch 32.
  ARGS=(dataset=qm9 dataset.remove_h=True train.n_epochs=20 train.batch_size=32 general.name=t12_qm9_$VARIANT)
elif [ "$DATASET" = ego ]; then
  # Table 13: Ego, 100 epochs (configs/experiment/ego.yaml for the rest).
  ARGS=(+experiment=ego train.n_epochs=100 general.name=t13_ego_$VARIANT)
else
  echo "unknown dataset: $DATASET" >&2
  exit 1
fi

uv run python main.py "${ARGS[@]}" model.use_novel_sampling=$NOVEL general.wandb=offline
