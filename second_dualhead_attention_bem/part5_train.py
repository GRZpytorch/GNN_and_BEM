import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from part3_dataset import build_loaders
from part4_model import DualHeadGCNVAE, compute_losses


def evaluate(model, loader, device, beta_kl: float = 1e-5):
    """
    在验证集或测试集上评估模型。

    参数:
        model: 已构建好的神经网络模型
        loader: 数据加载器（如 val_loader 或 test_loader）
        device: 计算设备（cpu 或 cuda）
        beta_kl: VAE 的 KL 正则权重

    返回:
        avg: 一个字典，包含该数据集上的平均损失信息
             例如:
             {
                 "loss_total": ...,
                 "loss_sup": ...,
                 "loss_T": ...,
                 "loss_q": ...,
                 "loss_kl": ...,
                 "T_unknown_mae": ...,
                 "q_unknown_mae": ...
             }
    """
    model.eval()
    records = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # 新版模型前向传播，返回 ModelOutput
            output = model(data)

            # 新版损失：纯监督 + VAE KL 正则
            _, info, _ = compute_losses(
                data,
                output,
                beta_kl=beta_kl,
            )

            records.append(info)

    if not records:
        raise RuntimeError("evaluate() 收到空的 loader，无法计算评估结果。")

    avg = {k: sum(r[k] for r in records) / len(records) for k in records[0]}
    return avg


def train(
    dataset_root: str = "dataset_rect_bem_gnn",
    save_dir: str = "artifacts_train",
    epochs: int = 1500,
    lr: float = 1e-3,
    hidden_channels: int = 128,
    latent_channels: int = 32,
    decoder_channels: int = 128,
    dropout: float = 0.05,
    use_variational: bool = True,
    beta_kl: float = 1e-5,
):
    """
    单图训练主函数（新版：纯监督 + VAE 正则）。

    当前设定说明:
        - 数据集通常是固定边值问题生成的单图样本
        - train / val / test 中可能各只有一个图
        - 训练本质上更接近“单图上的监督拟合 + KL 正则化”

    参数:
        dataset_root: 数据集根目录
        save_dir: 模型、曲线、日志保存目录
        epochs: 训练轮数
        lr: 学习率
        hidden_channels: 编码器隐藏层维度
        latent_channels: VAE 潜变量维度
        decoder_channels: 解码器隐藏层维度
        dropout: dropout 概率
        use_variational: 是否启用 VAE 重参数化
        beta_kl: KL 损失权重
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = build_loaders(dataset_root, batch_size=1)

    if len(train_loader) == 0:
        raise RuntimeError("train_loader 为空，请检查数据集是否正确生成。")
    if len(val_loader) == 0:
        raise RuntimeError("val_loader 为空，请检查数据集是否正确生成。")

    # 新版模型
    model = DualHeadGCNVAE(
        in_channels=9,
        hidden_channels=hidden_channels,
        latent_channels=latent_channels,
        decoder_channels=decoder_channels,
        dropout=dropout,
        use_variational=use_variational,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 新版训练历史
    history = {
        "epoch": [],
        "train_total": [],
        "train_sup": [],
        "train_T": [],
        "train_q": [],
        "train_kl": [],
        "train_T_unknown_mae": [],
        "train_q_unknown_mae": [],
        "val_total": [],
        "val_sup": [],
        "val_T": [],
        "val_q": [],
        "val_kl": [],
        "val_T_unknown_mae": [],
        "val_q_unknown_mae": [],
    }

    best_val = float("inf")
    best_path = save_path / "best_model.pt"
    last_path = save_path / "last_model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_records = []

        for data in train_loader:
            data = data.to(device)

            optimizer.zero_grad()

            # 前向传播：返回 ModelOutput
            output = model(data)

            # 新版损失
            loss, info, _ = compute_losses(
                data,
                output,
                beta_kl=beta_kl,
            )

            loss.backward()
            optimizer.step()

            train_records.append(info)

        if not train_records:
            raise RuntimeError("训练过程中 train_records 为空，请检查 train_loader。")

        train_avg = {k: sum(r[k] for r in train_records) / len(train_records) for k in train_records[0]}

        val_avg = evaluate(
            model,
            val_loader,
            device,
            beta_kl=beta_kl,
        )

        history["epoch"].append(epoch)

        history["train_total"].append(train_avg["loss_total"])
        history["train_sup"].append(train_avg["loss_sup"])
        history["train_T"].append(train_avg["loss_T"])
        history["train_q"].append(train_avg["loss_q"])
        history["train_kl"].append(train_avg["loss_kl"])
        history["train_T_unknown_mae"].append(train_avg["T_unknown_mae"])
        history["train_q_unknown_mae"].append(train_avg["q_unknown_mae"])

        history["val_total"].append(val_avg["loss_total"])
        history["val_sup"].append(val_avg["loss_sup"])
        history["val_T"].append(val_avg["loss_T"])
        history["val_q"].append(val_avg["loss_q"])
        history["val_kl"].append(val_avg["loss_kl"])
        history["val_T_unknown_mae"].append(val_avg["T_unknown_mae"])
        history["val_q_unknown_mae"].append(val_avg["q_unknown_mae"])

        if val_avg["loss_total"] < best_val:
            best_val = val_avg["loss_total"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "in_channels": 9,
                        "hidden_channels": hidden_channels,
                        "latent_channels": latent_channels,
                        "decoder_channels": decoder_channels,
                        "dropout": dropout,
                        "use_variational": use_variational,
                    },
                    "best_val_loss": best_val,
                    "epoch": epoch,
                    "beta_kl": beta_kl,
                },
                best_path,
            )

        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch {epoch:4d} | "
                f"train_total={train_avg['loss_total']:.6e} | "
                f"val_total={val_avg['loss_total']:.6e} | "
                f"val_sup={val_avg['loss_sup']:.6e} | "
                f"val_kl={val_avg['loss_kl']:.6e} | "
                f"val_T_mae={val_avg['T_unknown_mae']:.6e} | "
                f"val_q_mae={val_avg['q_unknown_mae']:.6e}"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "in_channels": 9,
                "hidden_channels": hidden_channels,
                "latent_channels": latent_channels,
                "decoder_channels": decoder_channels,
                "dropout": dropout,
                "use_variational": use_variational,
            },
            "epoch": epochs,
            "beta_kl": beta_kl,
        },
        last_path,
    )

    with open(save_path / "loss_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 画新版损失曲线
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_total"], label="train_total")
    plt.plot(history["epoch"], history["val_total"], label="val_total")
    plt.plot(history["epoch"], history["train_sup"], label="train_sup", linestyle="--")
    plt.plot(history["epoch"], history["val_sup"], label="val_sup", linestyle="-.")
    plt.plot(history["epoch"], history["train_kl"], label="train_kl", linestyle=":")
    plt.plot(history["epoch"], history["val_kl"], label="val_kl", linestyle=":")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "loss_curve.png", dpi=200)
    plt.close()

    # 再画一个 MAE 曲线，便于看未知量预测质量
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_T_unknown_mae"], label="train_T_unknown_mae")
    plt.plot(history["epoch"], history["val_T_unknown_mae"], label="val_T_unknown_mae")
    plt.plot(history["epoch"], history["train_q_unknown_mae"], label="train_q_unknown_mae", linestyle="--")
    plt.plot(history["epoch"], history["val_q_unknown_mae"], label="val_q_unknown_mae", linestyle="--")
    plt.xlabel("epoch")
    plt.ylabel("MAE")
    plt.yscale("log")
    plt.title("Unknown Boundary MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "mae_curve.png", dpi=200)
    plt.close()

    summary = {
        "best_val_loss": best_val,
        "best_model": str(best_path.resolve()),
        "last_model": str(last_path.resolve()),
        "epochs": epochs,
        "lr": lr,
        "hidden_channels": hidden_channels,
        "latent_channels": latent_channels,
        "decoder_channels": decoder_channels,
        "dropout": dropout,
        "use_variational": use_variational,
        "beta_kl": beta_kl,
        "device": str(device),
    }
    with open(save_path / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved best model to: {best_path.resolve()}")
    print(f"Saved last model to: {last_path.resolve()}")
    print(f"Saved loss curve to: {(save_path / 'loss_curve.png').resolve()}")
    print(f"Saved mae curve to: {(save_path / 'mae_curve.png').resolve()}")
    print(f"Best validation loss: {best_val:.6e}")


if __name__ == "__main__":
    train()
