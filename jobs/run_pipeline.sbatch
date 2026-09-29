#!/usr/bin/env bash
# Generic SLURM wrapper: runs one pipeline script with the given CLI args.
# Usually submitted by jobs/submit_all.sh, but can be used directly (from the repo root):
#   sbatch --job-name=poisson_SNO jobs/run_pipeline.sbatch pipelines/pipeline_poisson_fixed_grid_SNO.py \
#          --model SNO --gentype FCG --sdiv 1 --nrep 100
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch jobs/run_pipeline.sbatch <pipeline.py> [pipeline args...]" >&2
    exit 1
fi
PIPELINE="$1"; shift

source "$REPO_ROOT/venv/bin/activate"
# Pipelines import solvers/, architectures/, transforms/ from the repo root.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# SLURM already restricts the job to its allocated GPU; pass that id through so the
# pipeline's own CUDA_VISIBLE_DEVICES override points at the same device.
GPU="${CUDA_VISIBLE_DEVICES:-0}"
GPU="${GPU%%,*}"

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} gpu=$GPU"
echo "cmd: python $PIPELINE $* --cuda $GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

python -u "$PIPELINE" "$@" --cuda "$GPU"
