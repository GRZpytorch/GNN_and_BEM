from __future__ import annotations

"""Training loop, validation, checkpointing, and learning-curve visualization."""


import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from dataset import BoundaryFluxDataset
from model import FluxGNN, loss_function


DEFAULT_CONFIG = {
    "data_dir": "data/case1",
    "case": "case1",

    # Use a dedicated output directory so previous checkpoints are not overwritten.
    "run_dir": "runs/case1_full_attention",

    "epochs": 1500,
    "batch_size": 1,
    "hidden_dim": 128,
    "latent_dim": 32,
    "dropout": 0.05,

    "lr": 1e-3,
    "weight_decay": 0.0,

    # VAE KL-divergence coefficient.
    "beta": 1e-3,

    # Gradient clipping threshold.
    "grad_clip": 1.0,

    "seed": 1234,
    "num_workers": 0,
}


def set_random_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Exact reproducibility across different hardware, PyTorch versions,
    and CUDA environments is not guaranteed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """
    Initialize the random state of each DataLoader worker.
    """
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@torch.no_grad()
def evaluate(
    model: FluxGNN,
    loader: DataLoader,
    device: torch.device,
    beta: float,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_kl = 0.0
    count = 0

    for data in loader:
        data = data.to(device)

        output = model(data)

        _, metrics = loss_function(
            output,
            data.y,
            beta=beta,
        )

        total_loss += metrics["loss"]
        total_mse += metrics["mse"]
        total_kl += metrics["kl"]
        count += 1

    denominator = max(count, 1)

    return {
        "loss": total_loss / denominator,
        "mse": total_mse / denominator,
        "kl": total_kl / denominator,
    }


def plot_history(
    history_path: Path,
    out_path: Path,
) -> None:
    import pandas as pd

    df = pd.read_csv(history_path)

    plt.figure(figsize=(7, 4.5))

    plt.plot(
        df["epoch"],
        df["train_loss"],
        label="Train total loss",
    )

    plt.plot(
        df["epoch"],
        df["val_loss"],
        label="Validation total loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_loss_components(
    history_path: Path,
    out_path: Path,
) -> None:
    """
    Plot reconstruction and KL losses separately.
    """
    import pandas as pd

    df = pd.read_csv(history_path)

    plt.figure(figsize=(7, 4.5))

    plt.plot(
        df["epoch"],
        df["train_mse"],
        label="Train MSE",
    )

    plt.plot(
        df["epoch"],
        df["val_mse"],
        label="Validation MSE",
    )

    plt.plot(
        df["epoch"],
        df["train_kl"],
        label="Train KL",
    )

    plt.plot(
        df["epoch"],
        df["val_kl"],
        label="Validation KL",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss component")
    plt.title("Reconstruction and KL Losses")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def train_model(
    data_dir: str | Path,
    case: str,
    run_dir: str | Path,
    epochs: int = 1500,
    batch_size: int = 1,
    hidden_dim: int = 128,
    latent_dim: int = 32,
    dropout: float = 0.05,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    beta: float = 1e-3,
    grad_clip: float = 1.0,
    seed: int = 1234,
    num_workers: int = 0,
) -> None:
    data_dir = Path(data_dir)
    run_dir = Path(run_dir)

    if epochs <= 0:
        raise ValueError("epochs must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")

    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")

    if beta < 0:
        raise ValueError("beta must be non-negative")

    if grad_clip < 0:
        raise ValueError("grad_clip must be non-negative")

    set_random_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    train_set = BoundaryFluxDataset(
        data_dir,
        split="train",
        full_load=False,
    )

    val_set = BoundaryFluxDataset(
        data_dir,
        split="val",
        full_load=False,
    )

    if len(train_set) == 0:
        raise ValueError("The training dataset is empty")

    if len(val_set) == 0:
        raise ValueError("The validation dataset is empty")

    sample = train_set.get(0)

    if sample.x.ndim != 2:
        raise ValueError(
            f"Expected sample.x to have shape [N, F], "
            f"but received {tuple(sample.x.shape)}"
        )

    if not hasattr(sample, "edge_index"):
        raise ValueError("Dataset samples do not contain edge_index")

    if not hasattr(sample, "y"):
        raise ValueError("Dataset samples do not contain target y")

    in_dim = sample.x.shape[1]

    # An independent generator makes training-set shuffling reproducible.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=loader_generator,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        pin_memory=torch.cuda.is_available(),
    )

    model = FluxGNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_path = run_dir / "history.csv"
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"

    curve_path = run_dir / "loss_curve.png"
    component_curve_path = run_dir / "loss_components.png"

    best_val = float("inf")

    history_columns = [
        "epoch",
        "train_loss",
        "train_mse",
        "train_kl",
        "val_loss",
        "val_mse",
        "val_kl",
    ]

    with history_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(history_columns)

        for epoch in range(1, epochs + 1):
            model.train()

            total_loss = 0.0
            total_mse = 0.0
            total_kl = 0.0
            batch_count = 0

            for data in train_loader:
                data = data.to(
                    device,
                    non_blocking=torch.cuda.is_available(),
                )

                optimizer.zero_grad(set_to_none=True)

                output = model(data)

                loss, metrics = loss_function(
                    output,
                    data.y,
                    beta=beta,
                )

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss detected at epoch {epoch}: "
                        f"{float(loss.detach().cpu())}"
                    )

                loss.backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip,
                    )

                optimizer.step()

                total_loss += metrics["loss"]
                total_mse += metrics["mse"]
                total_kl += metrics["kl"]
                batch_count += 1

            denominator = max(batch_count, 1)

            train_metrics = {
                "loss": total_loss / denominator,
                "mse": total_mse / denominator,
                "kl": total_kl / denominator,
            }

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                beta=beta,
            )

            writer.writerow(
                [
                    epoch,
                    train_metrics["loss"],
                    train_metrics["mse"],
                    train_metrics["kl"],
                    val_metrics["loss"],
                    val_metrics["mse"],
                    val_metrics["kl"],
                ]
            )

            file.flush()

            state = {
                "case": case,
                "in_dim": in_dim,
                "hidden_dim": hidden_dim,
                "latent_dim": latent_dim,
                "dropout": dropout,
                "beta": beta,
                "grad_clip": grad_clip,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_metrics["loss"],
                "train_mse": train_metrics["mse"],
                "train_kl": train_metrics["kl"],
                "val_loss": val_metrics["loss"],
                "val_mse": val_metrics["mse"],
                "val_kl": val_metrics["kl"],
            }

            torch.save(state, last_path)

            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(state, best_path)

            if epoch % 50 == 0 or epoch == 1:
                print(
                    f"epoch={epoch:04d} "
                    f"train={train_metrics['loss']:.6e} "
                    f"train_mse={train_metrics['mse']:.6e} "
                    f"train_kl={train_metrics['kl']:.6e} "
                    f"val={val_metrics['loss']:.6e} "
                    f"val_mse={val_metrics['mse']:.6e} "
                    f"val_kl={val_metrics['kl']:.6e}"
                )

    plot_history(
        history_path,
        curve_path,
    )

    plot_loss_components(
        history_path,
        component_curve_path,
    )

    print(f"Training finished. Best val={best_val:.6e}")
    print(f"Saved checkpoints and loss curves to {run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path(DEFAULT_CONFIG["data_dir"]),
    )

    parser.add_argument(
        "--case",
        type=str,
        default=DEFAULT_CONFIG["case"],
    )

    parser.add_argument(
        "--run_dir",
        type=Path,
        default=Path(DEFAULT_CONFIG["run_dir"]),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["epochs"],
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=DEFAULT_CONFIG["hidden_dim"],
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        default=DEFAULT_CONFIG["latent_dim"],
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_CONFIG["dropout"],
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_CONFIG["lr"],
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=DEFAULT_CONFIG["weight_decay"],
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_CONFIG["beta"],
    )

    parser.add_argument(
        "--grad_clip",
        type=float,
        default=DEFAULT_CONFIG["grad_clip"],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG["seed"],
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_CONFIG["num_workers"],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_model(
        data_dir=args.data_dir,
        case=args.case,
        run_dir=args.run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta=args.beta,
        grad_clip=args.grad_clip,
        seed=args.seed,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
