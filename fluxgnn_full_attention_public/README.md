# FluxGNN: Full-Graph Attention VAE for Boundary-Flux Reconstruction

FluxGNN is a research implementation that combines a boundary element method
(BEM) with a variational graph neural network for reconstructing boundary
Neumann data in two-dimensional Poisson/Laplace problems.

The graph model operates on boundary elements. Each boundary node exchanges
information with every other boundary node through a learned multi-head
attention operator, allowing the network to represent the nonlocal interactions
that are characteristic of boundary-integral formulations.

## Method Overview

```text
Boundary geometry and physical features
                 |
                 v
      Fully connected directed graph
                 |
                 v
      Full-Graph Attention Encoder
            (2 layers)
                 |
            mu, logvar
                 |
                 v
           latent variable z
                 |
          concatenate [x, z]
                 |
                 v
      Full-Graph Attention Decoder
            (3 layers)
                 |
                 v
        predicted boundary flux q
                 |
                 v
       BEM interior reconstruction
```

The model uses a node-wise variational latent representation. During training,
the latent state is sampled with the reparameterization trick. During
evaluation, the latent mean is used to obtain deterministic predictions.

## Attention Operator

For every directed source-target edge, the model computes multi-head query,
key, and value projections. The base query-key score is augmented by a learned
pairwise relation term constructed from

```text
[source features, target features, target - source]
```

A second learned relation network gates the value message for each attention
head. Incoming messages are normalized independently for each target node. A
residual projection preserves node-local information.

The boundary graphs are fully connected and omit self-edges by default because
the residual path already retains the node's own state.

## Node Features

The generated node feature matrix preserves the following order:

```text
[x, y, nx, ny, ds, u_bc, f_boundary, problem_parameters...]
```

where:

- `x, y` are boundary collocation coordinates;
- `nx, ny` are outward unit-normal components;
- `ds` is the boundary-element length;
- `u_bc` is the exact Dirichlet boundary value supplied to the model;
- `f_boundary` is the source term evaluated on the boundary;
- `problem_parameters` contains case-specific global parameters repeated at
  every boundary node.

The learning target is the exact outward normal derivative `q`.

## Benchmark Problems

The repository defines three analytical benchmark cases:

- `case1`: constant-source Poisson problem on the unit disk;
- `case2`: an analytical problem on the unit disk;
- `case3`: an analytical problem on the unit square.

The case registry in `problems.py` is the authoritative source for the exact
field, source term, flux definition, and sampled parameters of each benchmark.

## Repository Layout

```text
.
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── requirements.txt
├── .gitignore
├── bem_solver.py
├── dataset.py
├── generate_data.py
├── geometry.py
├── model.py
├── problems.py
├── train.py
├── test.py
└── run_pipeline.py
```

### `bem_solver.py`

Implements the two-dimensional fundamental solution, source-normal derivative,
boundary-panel quadrature, cached collocation solve, and interior-field
evaluation.

### `geometry.py`

Builds disk and square boundary discretizations, fully connected directed
graphs, interior grids, and domain quadrature rules.

### `problems.py`

Defines analytical benchmark solutions, source terms, exact boundary fluxes,
and case-specific parameters.

### `generate_data.py`

Generates the graph data stored in `train`, `val`, and `test` directories.

### `dataset.py`

Loads generated NPZ samples into PyTorch Geometric `Data` objects.

### `model.py`

Defines `FullGraphAttentionConv`, the variational `FluxGNN` architecture, and
the reconstruction-plus-KL training objective.

### `train.py`

Provides deterministic seeding, training and validation loops, checkpointing,
CSV history output, and loss plots.

### `test.py`

Loads a checkpoint, evaluates boundary-flux error, performs BEM reconstruction,
computes interior-field metrics, and generates publication-oriented figures.

### `run_pipeline.py`

Runs dataset generation, training, and testing in sequence using a single
case-specific configuration.

## Installation

Python 3.10 or newer is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

For CUDA execution, install the PyTorch build appropriate for the local CUDA
toolkit before installing the remaining packages.

## Quick Start

Generate a dataset:

```bash
python generate_data.py --case case3 --out-dir data/case3
```

Train the attention model:

```bash
python train.py --data-dir data/case3 --case case3 --run-dir runs/case3_full_attention
```

Evaluate the best checkpoint:

```bash
python test.py \
  --data-dir data/case3 \
  --case case3 \
  --ckpt runs/case3_full_attention/best.pt \
  --out-dir results/case3_full_attention
```

The repository also provides an end-to-end runner:

```bash
python run_pipeline.py
```

The default case and training settings used by the pipeline can be edited near
the top of `run_pipeline.py`.

## Default Training Configuration

The default pipeline uses:

| Parameter | Value |
| --- | ---: |
| Epochs | 1500 |
| Batch size | 1 |
| Hidden dimension | 128 |
| Latent dimension | 32 |
| Dropout | 0.05 |
| Learning rate | 1e-3 |
| Weight decay | 0 |
| KL coefficient | 1e-3 |
| Gradient clipping | 1.0 |
| Random seed | 1234 |

## Dataset Scope

The current generator intentionally creates one physical sample and writes the
same sample to the training and validation splits. The test split uses the same
physical problem while additionally storing analytical interior-field data.

This setup should therefore be interpreted as a fixed-problem reconstruction
or fitting experiment, not as evidence of generalization to unseen physical
samples. Generalization studies should generate independent parameter,
geometry, or boundary-condition samples across the data splits.

## Reproducibility

The training script seeds Python, NumPy, and PyTorch. Exact bitwise
reproducibility is not guaranteed across different hardware, CUDA versions, or
PyTorch releases.

Generated datasets, checkpoints, logs, and evaluation results are excluded from
Git by default.

## Numerical Conventions

The BEM implementation documents its fundamental-solution and normal-derivative
sign conventions directly in `bem_solver.py`. New problem definitions should
use the same convention for the exact outward normal derivative.

## License

This repository is released under the MIT License. Replace `The Authors` in
`LICENSE` with the preferred author or organization name before a formal
release.

## Citation

If this software supports a publication, add the corresponding paper citation
or DOI here before the archival release.
