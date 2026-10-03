"""Train the dual-head attention VAE and save checkpoints and learning curves."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from part3_dataset import build_loaders
from part4_model import DualHeadGCNVAE, compute_losses

def evaluate(model, loader, device, beta_kl: float=1e-05):
    model.eval()
    records = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            _, info, _ = compute_losses(data, output, beta_kl=beta_kl)
            records.append(info)
    if not records:
        raise RuntimeError('evaluate() received an empty data loader.')
    avg = {k: sum((r[k] for r in records)) / len(records) for k in records[0]}
    return avg

def train(dataset_root: str='dataset_rect_bem_gnn', save_dir: str='artifacts_train', epochs: int=1500, lr: float=0.001, hidden_channels: int=128, latent_channels: int=32, decoder_channels: int=128, dropout: float=0.05, use_variational: bool=True, beta_kl: float=1e-05):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, _ = build_loaders(dataset_root, batch_size=1)
    if len(train_loader) == 0:
        raise RuntimeError('train_loader is empty. Check the dataset.')
    if len(val_loader) == 0:
        raise RuntimeError('val_loader is empty. Check the dataset.')
    model = DualHeadGCNVAE(in_channels=9, hidden_channels=hidden_channels, latent_channels=latent_channels, decoder_channels=decoder_channels, dropout=dropout, use_variational=use_variational).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {'epoch': [], 'train_total': [], 'train_sup': [], 'train_T': [], 'train_q': [], 'train_kl': [], 'train_T_unknown_mae': [], 'train_q_unknown_mae': [], 'val_total': [], 'val_sup': [], 'val_T': [], 'val_q': [], 'val_kl': [], 'val_T_unknown_mae': [], 'val_q_unknown_mae': []}
    best_val = float('inf')
    best_path = save_path / 'best_model.pt'
    last_path = save_path / 'last_model.pt'
    for epoch in range(1, epochs + 1):
        model.train()
        train_records = []
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss, info, _ = compute_losses(data, output, beta_kl=beta_kl)
            loss.backward()
            optimizer.step()
            train_records.append(info)
        if not train_records:
            raise RuntimeError('No training records were produced. Check train_loader.')
        train_avg = {k: sum((r[k] for r in train_records)) / len(train_records) for k in train_records[0]}
        val_avg = evaluate(model, val_loader, device, beta_kl=beta_kl)
        history['epoch'].append(epoch)
        history['train_total'].append(train_avg['loss_total'])
        history['train_sup'].append(train_avg['loss_sup'])
        history['train_T'].append(train_avg['loss_T'])
        history['train_q'].append(train_avg['loss_q'])
        history['train_kl'].append(train_avg['loss_kl'])
        history['train_T_unknown_mae'].append(train_avg['T_unknown_mae'])
        history['train_q_unknown_mae'].append(train_avg['q_unknown_mae'])
        history['val_total'].append(val_avg['loss_total'])
        history['val_sup'].append(val_avg['loss_sup'])
        history['val_T'].append(val_avg['loss_T'])
        history['val_q'].append(val_avg['loss_q'])
        history['val_kl'].append(val_avg['loss_kl'])
        history['val_T_unknown_mae'].append(val_avg['T_unknown_mae'])
        history['val_q_unknown_mae'].append(val_avg['q_unknown_mae'])
        if val_avg['loss_total'] < best_val:
            best_val = val_avg['loss_total']
            torch.save({'model_state_dict': model.state_dict(), 'config': {'in_channels': 9, 'hidden_channels': hidden_channels, 'latent_channels': latent_channels, 'decoder_channels': decoder_channels, 'dropout': dropout, 'use_variational': use_variational}, 'best_val_loss': best_val, 'epoch': epoch, 'beta_kl': beta_kl}, best_path)
        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:4d} | train_total={train_avg['loss_total']:.6e} | val_total={val_avg['loss_total']:.6e} | val_sup={val_avg['loss_sup']:.6e} | val_kl={val_avg['loss_kl']:.6e} | val_T_mae={val_avg['T_unknown_mae']:.6e} | val_q_mae={val_avg['q_unknown_mae']:.6e}")
    torch.save({'model_state_dict': model.state_dict(), 'config': {'in_channels': 9, 'hidden_channels': hidden_channels, 'latent_channels': latent_channels, 'decoder_channels': decoder_channels, 'dropout': dropout, 'use_variational': use_variational}, 'epoch': epochs, 'beta_kl': beta_kl}, last_path)
    with open(save_path / 'loss_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    plt.figure(figsize=(8, 5))
    plt.plot(history['epoch'], history['train_total'], label='train_total')
    plt.plot(history['epoch'], history['val_total'], label='val_total')
    plt.plot(history['epoch'], history['train_sup'], label='train_sup', linestyle='--')
    plt.plot(history['epoch'], history['val_sup'], label='val_sup', linestyle='-.')
    plt.plot(history['epoch'], history['train_kl'], label='train_kl', linestyle=':')
    plt.plot(history['epoch'], history['val_kl'], label='val_kl', linestyle=':')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / 'loss_curve.png', dpi=200)
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(history['epoch'], history['train_T_unknown_mae'], label='train_T_unknown_mae')
    plt.plot(history['epoch'], history['val_T_unknown_mae'], label='val_T_unknown_mae')
    plt.plot(history['epoch'], history['train_q_unknown_mae'], label='train_q_unknown_mae', linestyle='--')
    plt.plot(history['epoch'], history['val_q_unknown_mae'], label='val_q_unknown_mae', linestyle='--')
    plt.xlabel('epoch')
    plt.ylabel('MAE')
    plt.yscale('log')
    plt.title('Unknown Boundary MAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / 'mae_curve.png', dpi=200)
    plt.close()
    summary = {'best_val_loss': best_val, 'best_model': str(best_path.resolve()), 'last_model': str(last_path.resolve()), 'epochs': epochs, 'lr': lr, 'hidden_channels': hidden_channels, 'latent_channels': latent_channels, 'decoder_channels': decoder_channels, 'dropout': dropout, 'use_variational': use_variational, 'beta_kl': beta_kl, 'device': str(device)}
    with open(save_path / 'train_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'Saved best model to: {best_path.resolve()}')
    print(f'Saved last model to: {last_path.resolve()}')
    print(f"Saved loss curve to: {(save_path / 'loss_curve.png').resolve()}")
    print(f"Saved mae curve to: {(save_path / 'mae_curve.png').resolve()}")
    print(f'Best validation loss: {best_val:.6e}')
if __name__ == '__main__':
    train()
