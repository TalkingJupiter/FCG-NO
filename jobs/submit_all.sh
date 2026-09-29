#!/usr/bin/env bash
# Submits every experiment needed to reproduce the FCG-NO paper results
# (Rudikov et al., ICML 2024, arXiv:2402.05598). Each job loops over the grids
# hard-coded in its pipeline (32/64/128 for fixed-grid runs).
#
# Usage (from anywhere):
#   jobs/submit_all.sh                # submit everything
#   jobs/submit_all.sh --dry-run      # only print the sbatch commands
#   jobs/submit_all.sh main ablation  # submit selected groups: main, ablation, diffgrid
#
# Hyperparameters match the notebooks: samples_div=1 (N_train = grid), N_repeats=100
# Krylov iterations per training RHS, m_max=20 (set inside the pipelines).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SDIV=1
NREP=100
DRY_RUN=0
SELECTED=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        main|ablation|diffgrid) SELECTED+=("$arg") ;;
        *) echo "unknown argument: $arg" >&2; exit 1 ;;
    esac
done
[[ ${#SELECTED[@]} -eq 0 ]] && SELECTED=(main ablation diffgrid)

[[ -f .env ]] || { echo "missing .env (needs RESULTS=...)" >&2; exit 1; }
[[ -d venv ]] || { echo "missing venv/ in repo root" >&2; exit 1; }
mkdir -p logs

submit() {
    local name="$1"; shift
    local cmd=(sbatch --job-name="$name" jobs/run_pipeline.sbatch "$@")
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "${cmd[*]}"
    else
        echo -n "$name: "
        "${cmd[@]}"
    fi
}

want() { [[ " ${SELECTED[*]} " == *" $1 "* ]]; }

for eq in poisson diffusion; do
    # Main comparison: each architecture as an FCG preconditioner, trained with the
    # Notay loss on Krylov (FCG) residuals, plus the unpreconditioned FCG baseline.
    if want main; then
        for arch in SNO FNO UNet DilResNet; do
            submit "${eq}_${arch}_FCG" "pipelines/pipeline_${eq}_fixed_grid_${arch}.py" \
                --model "$arch" --gentype FCG --sdiv $SDIV --nrep $NREP
        done
        submit "${eq}_Id" "pipelines/pipeline_${eq}_fixed_grid_SNO.py" \
            --model Id --gentype FCG --sdiv $SDIV --nrep $NREP
    fi

    # Ablation: random residuals r = f - A u0 instead of Krylov residuals.
    if want ablation; then
        submit "${eq}_SNO_random" "pipelines/pipeline_${eq}_fixed_grid_SNO.py" \
            --model SNO --gentype random --sdiv $SDIV --nrep $NREP
    fi

    # Cross-resolution: train SNO on a coarse grid, test on the 2x finer grid.
    # (Poisson: train 128 -> test 256; diffusion: 32 -> 64 and 64 -> 128.
    #  The diffusion pipeline ignores --nsamp and uses N_train = grid.)
    if want diffgrid; then
        submit "${eq}_SNO_diffgrid" "pipelines/pipeline_${eq}_diff_grids_SNO.py" \
            --model SNO --gentype FCG --nsamp 128 --nrep $NREP
    fi
done
