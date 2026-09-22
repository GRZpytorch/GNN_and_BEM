from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from pathlib import Path

from part2_build_graph import save_dataset
from part5_train import train
from part6_test import run_test


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate the single-graph BEM dataset, train the dual-head "
            "full-graph-attention VAE, and run the final test."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset_rect_bem_gnn",
        help="Dataset directory.",
    )
    parser.add_argument(
        "--train-dir",
        default="artifacts_train",
        help="Training artifact/checkpoint directory.",
    )
    parser.add_argument(
        "--test-dir",
        default="artifacts_test",
        help="Test result directory.",
    )
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=32)
    parser.add_argument("--decoder-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--beta-kl", type=float, default=1e-5)
    parser.add_argument(
        "--deterministic-latent",
        action="store_true",
        help="Disable stochastic VAE reparameterization during training.",
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Reuse an existing dataset.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse artifacts_train/best_model.pt.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Do not run the final test/plotting stage.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    if not args.skip_data:
        print("\n[1/3] Generating dataset")
        save_dataset(str(dataset_root))
    else:
        print("\n[1/3] Dataset generation skipped")

    if not args.skip_train:
        print("\n[2/3] Training model")
        train(
            dataset_root=str(dataset_root),
            save_dir=str(train_dir),
            epochs=args.epochs,
            lr=args.lr,
            hidden_channels=args.hidden_channels,
            latent_channels=args.latent_channels,
            decoder_channels=args.decoder_channels,
            dropout=args.dropout,
            use_variational=not args.deterministic_latent,
            beta_kl=args.beta_kl,
        )
    else:
        print("\n[2/3] Training skipped")

    if not args.skip_test:
        print("\n[3/3] Testing model")
        run_test(
            dataset_root=str(dataset_root),
            model_dir=str(train_dir),
            out_dir=str(test_dir),
        )
    else:
        print("\n[3/3] Testing skipped")


if __name__ == "__main__":
    main()
