from __future__ import annotations

"""Dataset generation utilities for the analytical benchmark problems."""


import argparse
from pathlib import Path
from typing import Any

import numpy as np

from geometry import (
    BoundaryMesh,
    DomainGrid,
    make_domain_grid,
    make_unit_circle_boundary,
    make_unit_square_boundary,
)
from problems import CASE_REGISTRY, ProblemSample


DEFAULT_CONFIG = {
    "case": "case3",
    "out_dir": None,
    "n_boundary": 128,
    "n_per_side": 32,
    "domain_res": 121,
    "seed": 1234,
}


# =========================================================
# Validation utilities
# =========================================================

def _validate_case(case: str) -> str:
    if case not in CASE_REGISTRY:
        available_cases = ", ".join(
            sorted(CASE_REGISTRY.keys())
        )

        raise ValueError(
            f"Unknown case={case!r}. "
            f"Available cases: {available_cases}"
        )

    return case


def _validate_positive_integer(
    value: int,
    name: str,
    minimum: int = 1,
) -> int:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(
            f"{name} must be an integer"
        )

    value = int(value)

    if value < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}, "
            f"but received {value}"
        )

    return value


def _check_finite(
    name: str,
    values: np.ndarray,
) -> None:
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(
            f"{name} contains NaN or infinite values"
        )


def _evaluate_scalar_field(
    function,
    points: np.ndarray,
    name: str,
) -> np.ndarray:
    """
    Evaluate a scalar-valued function and return one value per point.
    """
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    values = np.asarray(
        function(points),
        dtype=np.float64,
    ).reshape(-1)

    if values.shape[0] != points.shape[0]:
        raise ValueError(
            f"{name} returned {values.shape[0]} values "
            f"for {points.shape[0]} points"
        )

    _check_finite(
        name,
        values,
    )

    return values


def _evaluate_boundary_flux(
    sample: ProblemSample,
    coords: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the exact outward normal derivative at boundary nodes.
    """
    values = np.asarray(
        sample.q_fn(coords, normals),
        dtype=np.float64,
    ).reshape(-1)

    if values.shape[0] != coords.shape[0]:
        raise ValueError(
            "sample.q_fn returned an invalid number "
            "of boundary values"
        )

    _check_finite(
        "sample.q_fn(boundary)",
        values,
    )

    return values


# =========================================================
# Geometry construction
# =========================================================

def build_boundary(
    case: str,
    n_boundary: int,
    n_per_side: int,
) -> BoundaryMesh:
    """
    Preserve the original case-to-geometry mapping:

        case1 -> unit circle
        case2 -> unit circle
        case3 -> unit square
    """
    case = _validate_case(case)

    if case in {"case1", "case2"}:
        return make_unit_circle_boundary(
            n_boundary=n_boundary,
        )

    if case == "case3":
        return make_unit_square_boundary(
            n_per_side=n_per_side,
        )

    raise RuntimeError(
        f"Unhandled case={case!r}"
    )


def _validate_boundary(
    boundary: BoundaryMesh,
) -> None:
    coords = np.asarray(
        boundary.coords,
    )

    normals = np.asarray(
        boundary.normals,
    )

    ds = np.asarray(
        boundary.ds,
    ).reshape(-1)

    edge_index = np.asarray(
        boundary.edge_index,
    )

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            "boundary.coords must have shape [N, 2]"
        )

    if normals.shape != coords.shape:
        raise ValueError(
            "boundary.normals must have the same shape "
            "as boundary.coords"
        )

    if ds.shape[0] != coords.shape[0]:
        raise ValueError(
            "boundary.ds must contain one value per node"
        )

    if np.any(ds <= 0.0):
        invalid_indices = np.where(
            ds <= 0.0
        )[0]

        raise ValueError(
            "Boundary-element lengths must be positive. "
            f"Invalid indices: {invalid_indices.tolist()}"
        )

    normal_norms = np.linalg.norm(
        normals,
        axis=1,
    )

    if np.any(normal_norms <= 1e-12):
        invalid_indices = np.where(
            normal_norms <= 1e-12
        )[0]

        raise ValueError(
            "Boundary normals must be nonzero. "
            f"Invalid indices: {invalid_indices.tolist()}"
        )

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            "boundary.edge_index must have shape [2, E]"
        )

    if edge_index.dtype.kind not in {"i", "u"}:
        raise TypeError(
            "boundary.edge_index must contain integer indices"
        )

    if edge_index.size > 0:
        if edge_index.min() < 0:
            raise ValueError(
                "boundary.edge_index contains negative indices"
            )

        if edge_index.max() >= coords.shape[0]:
            raise ValueError(
                "boundary.edge_index contains an index "
                "outside the valid node range"
            )


def _validate_domain_grid(
    domain_grid: DomainGrid,
) -> None:
    points = np.asarray(
        domain_grid.points,
    )

    mask = np.asarray(
        domain_grid.mask,
    ).reshape(-1)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            "domain_grid.points must have shape [M, 2]"
        )

    if mask.shape[0] != points.shape[0]:
        raise ValueError(
            "domain_grid.mask length does not match "
            "domain_grid.points"
        )

    if mask.dtype != np.bool_:
        raise TypeError(
            "domain_grid.mask must be a Boolean array"
        )

    if len(domain_grid.shape) != 2:
        raise ValueError(
            "domain_grid.shape must contain two dimensions"
        )

    if int(np.prod(domain_grid.shape)) != points.shape[0]:
        raise ValueError(
            f"domain_grid.shape={domain_grid.shape} does not match "
            f"the number of points {points.shape[0]}"
        )

    if not np.any(mask):
        raise ValueError(
            "domain_grid.mask does not contain any interior points"
        )


# =========================================================
# Sample construction
# =========================================================

def _make_save_dict(
    boundary: BoundaryMesh,
    sample: ProblemSample,
    domain_grid: DomainGrid,
    include_domain: bool,
) -> dict[str, Any]:
    """
    Build one saved sample.

    The original input and target definition is preserved:

        node input:
            coordinates,
            normals,
            element length,
            prescribed Dirichlet value,
            boundary source value,
            problem parameters.

        target:
            exact Neumann value q.
    """
    _validate_boundary(
        boundary
    )

    _validate_domain_grid(
        domain_grid
    )

    if sample.geometry_type != boundary.geometry_type:
        raise ValueError(
            f"Problem sample geometry_type={sample.geometry_type!r} "
            f"does not match boundary geometry_type="
            f"{boundary.geometry_type!r}"
        )

    coords = np.asarray(
        boundary.coords,
        dtype=np.float64,
    )

    normals = np.asarray(
        boundary.normals,
        dtype=np.float64,
    )

    ds = np.asarray(
        boundary.ds,
        dtype=np.float64,
    ).reshape(-1)

    edge_index = np.asarray(
        boundary.edge_index,
        dtype=np.int64,
    )

    num_nodes = coords.shape[0]

    # Exact boundary values.
    u_bc = _evaluate_scalar_field(
        sample.u_fn,
        coords,
        "sample.u_fn(boundary)",
    )

    q_exact = _evaluate_boundary_flux(
        sample=sample,
        coords=coords,
        normals=normals,
    )

    f_bnd = _evaluate_scalar_field(
        sample.f_fn,
        coords,
        "sample.f_fn(boundary)",
    )

    params = np.asarray(
        sample.params,
        dtype=np.float64,
    ).reshape(-1)

    _check_finite(
        "sample.params",
        params,
    )

    # Repeat global physical parameters at every boundary node.
    parameter_features = np.broadcast_to(
        params[None, :],
        (num_nodes, params.shape[0]),
    )

    # Preserve the original node-feature order:
    #
    # [x, y,
    #  nx, ny,
    #  ds,
    #  u_bc,
    #  f_bnd,
    #  parameters...]
    node_features = np.concatenate(
        [
            coords,
            normals,
            ds[:, None],
            u_bc[:, None],
            f_bnd[:, None],
            parameter_features,
        ],
        axis=1,
    )

    _check_finite(
        "node_features",
        node_features,
    )

    save_dict: dict[str, Any] = {
        "case_name": np.asarray(
            sample.case_name
        ),
        "geometry_type": np.asarray(
            sample.geometry_type
        ),

        "node_features": node_features.astype(
            np.float32
        ),

        "coords": coords.astype(
            np.float32
        ),
        "normals": normals.astype(
            np.float32
        ),
        "ds": ds.astype(
            np.float32
        ),

        "u_bc": u_bc.astype(
            np.float32
        ),
        "f_bnd": f_bnd.astype(
            np.float32
        ),
        "q_exact": q_exact.astype(
            np.float32
        ),

        "params": params.astype(
            np.float32
        ),

        "edge_index": edge_index.astype(
            np.int64
        ),
    }

    if include_domain:
        domain_points = np.asarray(
            domain_grid.points,
            dtype=np.float64,
        )

        domain_mask = np.asarray(
            domain_grid.mask,
            dtype=bool,
        ).reshape(-1)

        u_domain = _evaluate_scalar_field(
            sample.u_fn,
            domain_points,
            "sample.u_fn(domain)",
        )

        f_domain = _evaluate_scalar_field(
            sample.f_fn,
            domain_points,
            "sample.f_fn(domain)",
        )

        # evaluate_field() uses the interior representation formula.
        # Values outside the strict interior mask are excluded for every
        # geometry type, including square boundaries.
        u_domain = u_domain.copy()
        f_domain = f_domain.copy()

        u_domain[~domain_mask] = np.nan
        f_domain[~domain_mask] = np.nan

        save_dict.update(
            {
                "domain_points": domain_points.astype(
                    np.float32
                ),

                "domain_mask": domain_mask.astype(
                    np.bool_
                ),

                "domain_shape": np.asarray(
                    domain_grid.shape,
                    dtype=np.int64,
                ),

                "u_domain_exact": u_domain.astype(
                    np.float32
                ),

                "f_domain": f_domain.astype(
                    np.float32
                ),
            }
        )

    return save_dict


# =========================================================
# File utilities
# =========================================================

def _remove_old_samples(
    split_dir: Path,
) -> None:
    """
    Remove old generated samples.

    This is important when a directory previously contained multiple
    samples, while the current experiment intentionally uses one sample.
    """
    if not split_dir.exists():
        return

    for old_file in split_dir.glob(
        "sample_*.npz"
    ):
        old_file.unlink()


def _save_one(
    split_dir: Path,
    save_dict: dict[str, Any],
) -> Path:
    """
    Save one fixed sample using an atomic file replacement.
    """
    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        split_dir
        / "sample_000000.npz"
    )

    temporary_path = (
        split_dir
        / ".sample_000000.npz.tmp"
    )

    with temporary_path.open(
        "wb"
    ) as file:
        np.savez_compressed(
            file,
            **save_dict,
        )

    temporary_path.replace(
        output_path
    )

    return output_path


# =========================================================
# Dataset generation
# =========================================================

def generate_dataset(
    case: str,
    out_dir: str | Path,
    n_boundary: int = 128,
    n_per_side: int = 32,
    domain_res: int = 121,
    seed: int = 1234,
) -> None:
    """
    Generate one fixed physical sample.

    The same physical sample is deliberately used for:

        train/sample_000000.npz
        val/sample_000000.npz
        test/sample_000000.npz

    This setup is intended to verify whether the network can learn and
    reproduce one prescribed boundary mapping. It is not a generalization
    experiment.
    """
    case = _validate_case(
        case
    )

    n_boundary = _validate_positive_integer(
        n_boundary,
        "n_boundary",
        minimum=3,
    )

    n_per_side = _validate_positive_integer(
        n_per_side,
        "n_per_side",
        minimum=2,
    )

    domain_res = _validate_positive_integer(
        domain_res,
        "domain_res",
        minimum=3,
    )

    out_dir = Path(
        out_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        seed
    )

    # Generate exactly one physical sample.
    fixed_sample = CASE_REGISTRY[case](
        rng
    )

    boundary = build_boundary(
        case=case,
        n_boundary=n_boundary,
        n_per_side=n_per_side,
    )

    domain_grid = make_domain_grid(
        boundary.geometry_type,
        resolution=domain_res,
    )

    # Training and validation use the same fixed sample without
    # full-domain information.
    train_dict = _make_save_dict(
        boundary=boundary,
        sample=fixed_sample,
        domain_grid=domain_grid,
        include_domain=False,
    )

    # Testing uses the same physical sample and additionally stores
    # the analytical field over the domain.
    test_dict = _make_save_dict(
        boundary=boundary,
        sample=fixed_sample,
        domain_grid=domain_grid,
        include_domain=True,
    )

    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    test_dir = out_dir / "test"

    # Avoid residual files from previous experiments.
    _remove_old_samples(
        train_dir
    )
    _remove_old_samples(
        val_dir
    )
    _remove_old_samples(
        test_dir
    )

    train_path = _save_one(
        train_dir,
        train_dict,
    )

    val_path = _save_one(
        val_dir,
        train_dict,
    )

    test_path = _save_one(
        test_dir,
        test_dict,
    )

    print()
    print("Single-sample dataset generated successfully.")
    print(f"Case: {case}")
    print(f"Geometry: {boundary.geometry_type}")
    print(
        f"Boundary nodes: "
        f"{boundary.coords.shape[0]}"
    )
    print(
        f"Directed graph edges: "
        f"{boundary.edge_index.shape[1]}"
    )
    print(
        f"Input feature dimension: "
        f"{train_dict['node_features'].shape[1]}"
    )
    print(
        f"Parameters: "
        f"{fixed_sample.params.tolist()}"
    )
    print(f"Train sample: {train_path.resolve()}")
    print(f"Validation sample: {val_path.resolve()}")
    print(f"Test sample: {test_path.resolve()}")


# =========================================================
# Command-line interface
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one fixed sample for the "
            "single-problem GNN-BEM fitting experiment."
        )
    )

    parser.add_argument(
        "--case",
        type=str,
        default=DEFAULT_CONFIG["case"],
        choices=[
            "case1",
            "case2",
            "case3",
        ],
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_CONFIG["out_dir"],
        help=(
            "Output directory. When omitted, data/<case> is used."
        ),
    )

    parser.add_argument(
        "--n_boundary",
        type=int,
        default=DEFAULT_CONFIG["n_boundary"],
    )

    parser.add_argument(
        "--n_per_side",
        type=int,
        default=DEFAULT_CONFIG["n_per_side"],
    )

    parser.add_argument(
        "--domain_res",
        type=int,
        default=DEFAULT_CONFIG["domain_res"],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG["seed"],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_directory = (
        args.out_dir
        if args.out_dir is not None
        else Path("data") / args.case
    )

    generate_dataset(
        case=args.case,
        out_dir=output_directory,
        n_boundary=args.n_boundary,
        n_per_side=args.n_per_side,
        domain_res=args.domain_res,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
