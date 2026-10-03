from __future__ import annotations

"""End-to-end dataset generation, training, and evaluation pipeline."""


from pathlib import Path
from typing import Final, Literal

from generate_data import generate_dataset
from test import run_test
from train import train_model


# =========================================================
# Pipeline configuration
# =========================================================

CaseName = Literal["case1", "case2", "case3"]

CASE: Final[CaseName] = "case3"

# Rebuild the dataset even when it already exists.
REBUILD_DATA: Final[bool] = True

# Retrain the model even when best.pt already exists.
RETRAIN_MODEL: Final[bool] = True

# Test sample index.
TEST_SAMPLE_IDX: Final[int] = 0

# Domain quadrature resolution used during BEM evaluation.
QUAD_RES: Final[int] = 121


# =========================================================
# Training configuration
# =========================================================

TRAIN_CONFIG = {
    "epochs": 1500,
    "batch_size": 1,
    "hidden_dim": 128,
    "latent_dim": 32,
    "dropout": 0.05,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "beta": 1e-3,
    "grad_clip": 1.0,
    "seed": 1234,
    "num_workers": 0,
}


# =========================================================
# Case-specific paths
# =========================================================

CASE_PATHS = {
    "case1": {
        "data_dir": Path("data/case1"),
        "run_dir": Path("runs/case1_full_attention"),
        "result_dir": Path("results/case1_full_attention"),
    },
    "case2": {
        "data_dir": Path("data/case2"),
        "run_dir": Path("runs/case2_full_attention"),
        "result_dir": Path("results/case2_full_attention"),
    },
    "case3": {
        "data_dir": Path("data/case3"),
        "run_dir": Path("runs/case3_full_attention"),
        "result_dir": Path("results/case3_full_attention"),
    },
}


# =========================================================
# Validation utilities
# =========================================================

def validate_case(case: str) -> CaseName:
    if case not in CASE_PATHS:
        valid_cases = ", ".join(CASE_PATHS.keys())

        raise ValueError(
            f"Unsupported case '{case}'. "
            f"Expected one of: {valid_cases}"
        )

    return case  # type: ignore[return-value]


def split_has_samples(
    data_dir: Path,
    split: str,
) -> bool:
    """
    Return True when the selected split contains at least one NPZ sample.
    """
    split_dir = data_dir / split

    if not split_dir.is_dir():
        return False

    return next(
        split_dir.glob("sample_*.npz"),
        None,
    ) is not None


def dataset_exists(
    data_dir: str | Path,
) -> bool:
    """
    Check whether train, validation, and test splits all contain data.
    """
    data_dir = Path(data_dir)

    required_splits = (
        "train",
        "val",
        "test",
    )

    return all(
        split_has_samples(data_dir, split)
        for split in required_splits
    )


def checkpoint_exists(
    checkpoint_path: str | Path,
) -> bool:
    checkpoint_path = Path(checkpoint_path)

    return (
        checkpoint_path.is_file()
        and checkpoint_path.stat().st_size > 0
    )


# =========================================================
# Pipeline stages
# =========================================================

def prepare_dataset(
    case: CaseName,
    data_dir: Path,
) -> None:
    should_generate = (
        REBUILD_DATA
        or not dataset_exists(data_dir)
    )

    if not should_generate:
        print(
            f"Dataset already exists: {data_dir.resolve()}"
        )
        return

    print()
    print("=" * 60)
    print(f"Generating dataset for {case}")
    print(f"Output directory: {data_dir.resolve()}")
    print("=" * 60)

    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generate_dataset(
        case=case,
        out_dir=data_dir,
    )

    if not dataset_exists(data_dir):
        raise RuntimeError(
            "Dataset generation completed, but one or more required "
            f"splits are missing under {data_dir.resolve()}"
        )


def train_if_required(
    case: CaseName,
    data_dir: Path,
    run_dir: Path,
) -> Path:
    checkpoint_path = run_dir / "best.pt"

    should_train = (
        RETRAIN_MODEL
        or not checkpoint_exists(checkpoint_path)
    )

    if not should_train:
        print()
        print(
            f"Using existing checkpoint: "
            f"{checkpoint_path.resolve()}"
        )

        return checkpoint_path

    print()
    print("=" * 60)
    print(f"Training full-attention model for {case}")
    print(f"Run directory: {run_dir.resolve()}")
    print("=" * 60)

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_model(
        data_dir=data_dir,
        case=case,
        run_dir=run_dir,
        epochs=TRAIN_CONFIG["epochs"],
        batch_size=TRAIN_CONFIG["batch_size"],
        hidden_dim=TRAIN_CONFIG["hidden_dim"],
        latent_dim=TRAIN_CONFIG["latent_dim"],
        dropout=TRAIN_CONFIG["dropout"],
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        beta=TRAIN_CONFIG["beta"],
        grad_clip=TRAIN_CONFIG["grad_clip"],
        seed=TRAIN_CONFIG["seed"],
        num_workers=TRAIN_CONFIG["num_workers"],
    )

    if not checkpoint_exists(checkpoint_path):
        raise RuntimeError(
            "Training finished without producing a valid checkpoint: "
            f"{checkpoint_path.resolve()}"
        )

    return checkpoint_path


def test_model(
    case: CaseName,
    data_dir: Path,
    checkpoint_path: Path,
    result_dir: Path,
) -> None:
    # The current optimized test.py assumes:
    #
    #     complete Dirichlet boundary data -> complete Neumann prediction
    #
    # The mixed-boundary case1 experiment requires separate reconstruction
    # of the complete u and q vectors according to the information-type mask.
    if case == "case1":
        raise NotImplementedError(
            "The current test.py evaluates a complete "
            "Dirichlet-to-Neumann mapping and cannot directly evaluate "
            "the mixed-boundary case1 problem. A case1-specific test "
            "routine must first assemble the completed Dirichlet and "
            "Neumann boundary vectors using the boundary-information-type "
            "mask."
        )

    print()
    print("=" * 60)
    print(f"Testing model for {case}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Result directory: {result_dir.resolve()}")
    print("=" * 60)

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_test(
        data_dir=data_dir,
        case=case,
        ckpt_path=checkpoint_path,
        out_dir=result_dir,
        sample_idx=TEST_SAMPLE_IDX,
        quad_res=QUAD_RES,
    )


# =========================================================
# Main entry point
# =========================================================

def main() -> None:
    case = validate_case(CASE)

    paths = CASE_PATHS[case]

    data_dir = paths["data_dir"]
    run_dir = paths["run_dir"]
    result_dir = paths["result_dir"]

    print()
    print("GNN-BEM execution pipeline")
    print(f"Case: {case}")
    print(f"Rebuild data: {REBUILD_DATA}")
    print(f"Retrain model: {RETRAIN_MODEL}")

    prepare_dataset(
        case=case,
        data_dir=data_dir,
    )

    checkpoint_path = train_if_required(
        case=case,
        data_dir=data_dir,
        run_dir=run_dir,
    )

    test_model(
        case=case,
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        result_dir=result_dir,
    )

    print()
    print("=" * 60)
    print("Pipeline completed successfully.")
    print(f"Dataset: {data_dir.resolve()}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Results: {result_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
