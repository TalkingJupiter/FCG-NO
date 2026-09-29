# FCG-NO: Repository Notes and Reproduction Plan

## Paper

**Neural operators meet conjugate gradients: The FCG-NO method for efficient PDE solving**
Rudikov, Fanaskov, Muravleva, Laevsky, Oseledets. ICML 2024 (PMLR 235:42766–42782).

- arXiv: https://arxiv.org/abs/2402.05598
- Proceedings: https://proceedings.mlr.press/v235/rudikov24a.html
- ICML poster / slides: https://icml.cc/virtual/2024/poster/34406, https://icml.cc/media/icml-2024/Slides/34406_yZ8ykyC.pdf
- Upstream code: https://github.com/arudikov/FCG-NO

### Idea

Neural PDE solvers are usually not very accurate on their own. FCG-NO uses a neural operator as the
**preconditioner** inside Flexible Conjugate Gradient (FCG) instead. FCG allows a nonlinear
preconditioner, and Notay's theory guarantees convergence as long as the preconditioner
approximates A⁻¹ well enough in the energy norm. The network only needs to be roughly right, and
CG supplies the accuracy.

### Key design choices

- **Notay loss.** `E[ ||NO(r) − e||_A / ||e||_A ]`, where `e = A⁻¹r`. In the paper's ablations it
  needs 25–60% fewer iterations than a relative L2 loss.
- **Krylov training data.** Training residuals come from actually running CG/FCG from `u₀ ~ N(0, I)`,
  not from random vectors `r = f − A u₀`. Random residuals roughly double the iteration count at
  higher resolutions, and for diffusion they diverge at grid ≥ 64.
- **Spectral Neural Operator (SNO).** Encoder → processor → decoder in a Fourier basis (20 modes,
  4 layers). Because it works on basis coefficients, a model trained on a coarse grid can be reused
  on finer grids.

### Problems

- **Poisson:** a = 1, and f is a random trigonometric polynomial (5×5 coefficients).
- **Diffusion:** a is a random trigonometric polynomial + 10, and f is the same kind of random
  polynomial.
- Both use finite differences on the unit square with zero boundary values. Grids range from 32² to 256².

### Headline result

Stopping criterion: `||r||/||r₀|| ≤ 1e-6` on a 128² grid. FCG-NO takes about 20 iterations; GS(4)
takes about 78. These numbers come from an automated summary of the arXiv HTML, so check them
against the paper's tables before quoting them.

### Stated limitations

- SPD systems only, since the method is built on CG.
- Uniform grids only.
- Maximum resolution is limited by GPU memory.

## Repository layout

| Path | Contents |
|---|---|
| `solvers.py` | `FD_2D` (5-point finite-difference matrix for `−∇·(σ∇u)=f`, as a JAX `BCOO`), `FEM_2D`, and a scipy direct solver |
| `architectures/` | `fSNO`, `FNO`, `UNet`, `DilResNet`, `ChebNO`, all in Equinox |
| `transforms/` | Spectral analysis/synthesis operators (Fourier, Chebyshev, Legendre, …) used by SNO |
| `utilities.py` | Older matrix-free discretization helpers (poisson, helmholtz, divkrad, …); apparently unused by the pipelines |
| `pipelines/*.py` | 10 standalone experiment scripts: {poisson, diffusion} × {SNO, FNO, UNet, DilResNet} on a fixed grid, plus SNO on different grids |
| `notebooks/` | Notebook versions of the same experiments |
| `jobs/` | SLURM scripts (see below) |

Stack: JAX 0.4.30 (CUDA 12) with float64 enabled, Equinox, Optax.

### How a fixed-grid pipeline runs

Example: `pipeline_poisson_fixed_grid_SNO.py`.

1. Build `N = grid // sdiv` finite-difference systems with random right-hand sides (`PRNGKey(2)`).
2. Generate training data:
   - `--gentype FCG` runs unpreconditioned FCG for `nrep − 1` iterations and collects the
     normalized residual/error pairs.
   - `--gentype random` samples random residuals instead.
3. Train with the Notay loss. Optimizer: AdamW (weight decay 1e-2), learning rate 5e-4, halved
   every 50 epochs. Epochs: 150 for Poisson; 200 for diffusion (DilResNet uses 20).
4. Test: run FCG with the trained network as preconditioner (restart `m_max = 20`) on 20 new
   problems (`PRNGKey(12)`).
5. Save the residual history, the per-iteration Notay values and the pickled model to
   `$RESULTS/{Poisson,Elliptic}/<arch>/...`.

`--model Id` runs the unpreconditioned FCG baseline.

Each script loops over grids 32, 64 and 128. The different-grid pipelines behave differently:
- Poisson trains at 128 and tests at 256.
- Diffusion trains at 32 and 64 and tests at 2× each.

Output paths are read from `RESULTS` in `.env`. Scripts must be run from the repo root with the
repo root on `PYTHONPATH`.

## Reproduction jobs

- **`jobs/run_pipeline.sbatch`**: a generic wrapper. It requests 1× H100, 8 CPUs, 100G of memory
  and a 2-day limit, and writes logs to `logs/<name>_<jobid>.{out,err}`. It activates `venv`, sets
  `PYTHONPATH`, and passes the GPU SLURM allocated to `--cuda`.
- **`jobs/submit_all.sh [--dry-run] [main|ablation|diffgrid ...]`**: submits the experiments below.

Settings, taken from the notebooks: `sdiv = 1` (N_train = grid), `nrep = 100`, `m_max = 20`.

| Group | Jobs (for Poisson and diffusion) | Paper result |
|---|---|---|
| `main` | SNO, FNO, UNet, DilResNet with `--gentype FCG`, plus the `Id` baseline | Main comparison of architectures |
| `ablation` | SNO with `--gentype random` | Krylov vs random training data |
| `diffgrid` | SNO trained on a coarse grid, tested on the 2× finer grid | Reuse across resolutions |

That makes 14 jobs in total. The 2-day limit is a guess, so time one job before submitting everything.

### How to run

Prerequisites: `.env` sets `RESULTS=...`, and `venv/` is installed from `requirements.txt`. Both
already exist in this checkout.

1. Preview what will be submitted:
   ```bash
   cd /mnt/DISCL/work/bsencer/FCG-NO
   jobs/submit_all.sh --dry-run
   ```
2. Submit one small group first to check the runtime (2 jobs):
   ```bash
   jobs/submit_all.sh ablation
   squeue -u $USER
   tail -f logs/poisson_SNO_random_*.out
   ```
   If the 128-grid stage looks too slow for the 2-day limit, raise `--time` in
   `jobs/run_pipeline.sbatch` before continuing.
3. Submit the rest:
   ```bash
   jobs/submit_all.sh main diffgrid
   ```
   If you skipped step 2, run `jobs/submit_all.sh` instead to submit all 14 jobs.

### Outputs

Each run saves to `$RESULTS/{Poisson,Elliptic}/<arch>/`:
- `*_data.npz`, containing `R_model` (the residual history), `loss_model` (the per-iteration Notay
  values) and `history` (the training loss)
- `*_model`, the pickled network

The repo has no script that turns these outputs into the paper's tables. To build them, load
`R_model` and count the iterations until `||r_k|| / ||r_0|| ≤ 1e-6`, averaged over the 20 test
problems.

### Not covered by these jobs

- **Notay vs L2 loss ablation:** the L2 loss exists only as commented-out code in the pipelines.
  Running it needs a code change or a new flag.
- **Classical preconditioner baselines** (Jacobi, Gauss–Seidel, ILU): not implemented in this repo.
- **Cross-resolution setup:** the code doesn't match the paper exactly. It trains Poisson at
  128 → 256 and diffusion at 32 → 64 and 64 → 128. The paper describes training at 32 and testing
  at 128–256.

## Other issues noticed

- `requirements.txt` was missing `python-dotenv`, which the pipelines import. It has been added.
- `.gitignore` doesn't ignore `venv/`, `__pycache__/` or `.env`, so a `git add .` would commit them.
- The README's notebook and pipeline links point to `github.com/arudikov/FCG`; the repo is
  `arudikov/FCG-NO`.
- The diffusion different-grid pipeline ignores `--nsamp` and always uses `N_train = grid`.
- `jobs/pipeline_diffusion_diff_grids_SNO.sh` (the earlier job script) has no run command and a
  malformed `--error` line (`= logs/%x_j.out`). `run_pipeline.sbatch` replaces it.
