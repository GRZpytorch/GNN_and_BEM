# Dual-Head Full-Graph Attention VAE for BEM Boundary Reconstruction

This repository implements a graph-learning framework for reconstructing unknown boundary quantities in a two-dimensional Laplace boundary-value problem. A constant-element Boundary Element Method (BEM) provides the reference solution and interior-field reconstruction.

## Architecture

```text
Boundary node features
        |
        v
3 x Full-Graph Attention Encoder
        |
     mu, logvar
        |
        v
    latent z
        |
 concatenate [x, z]
        |
   +----+----+
   |         |
   v         v
3 x Attn   3 x Attn
T Decoder  q Decoder
   |         |
   v         v
Unknown T  Unknown q
```

Each boundary element uses the nine-dimensional feature vector:

```text
[x, y, nx, ny, length, is_T_known, is_q_known, known_T, known_q]
```

The attention operator combines multi-head query-key compatibility with learned source-target relation features `[source, target, target - source]`. A learned gate modulates each head, and a residual projection preserves node-local information.

## Repository Structure

```text
part1_bem_solver.py       BEM solver and reference problem
part2_build_graph.py      Graph and feature construction
part3_dataset.py          Dataset and data loaders
part4_model.py            Attention VAE and dual decoders
part5_train.py            Training pipeline
part6_test.py             Evaluation and BEM field reconstruction
run_all.py                End-to-end command-line entry point
examples/                 Optional visualization example
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU execution, install the PyTorch build appropriate for the local CUDA version before installing the remaining dependencies.

## Usage

Run the complete experiment:

```bash
python run_all.py
```

Run a short pipeline check:

```bash
python run_all.py --epochs 20
```

The stages can also be executed independently:

```bash
python part2_build_graph.py
python part5_train.py
python part6_test.py
```

Generated data, checkpoints, and figures are excluded from version control.

## Experimental Scope

The current dataset builder stores the same fixed graph in the training, validation, and test splits. The repository therefore demonstrates reconstruction on a fixed boundary-value problem rather than statistical generalization across independently sampled problems.

Generalization studies can extend the dataset builder to vary geometry, boundary-condition placement, boundary values, or other physical parameters.

## Checkpoint Compatibility

Checkpoints from the earlier GCN implementation are not parameter-compatible with this full-graph attention architecture. A new checkpoint must be trained after switching architectures.

## Citation

If this repository contributes to academic work, please cite the associated paper or project release when citation information is available.

## License

This project is distributed under the MIT License.
