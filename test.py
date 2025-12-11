# test.py

import torch
import numpy as np

from config import cfg
from dataset import prepare_dataloaders
from torch.utils.data import DataLoader
from models.unet2d import UNet

from visualize_results import (
    visualize_sample_images,
    visualize_test_windows_sec_original,
    visualize_test_windows_sec_normalized,
    # visualize_test_windows_sec,
    visualize_all_test_traffic,
    visualize_all_test_traffic_normalized,
    visualize_full_dataset_traffic,
    visualize_full_dataset_traffic_normalized,
)
from models import build_model_from_cfg

def compute_test_metrics(model, loader, device):
    """
    Compute MAE and MSE on test set in NORMALIZED scale.
    """
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    n_points = 0

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

            pred = model(X)
            diff = pred - Y

            mae_sum += diff.abs().sum().item()
            mse_sum += (diff ** 2).sum().item()
            n_points += diff.numel()

    mae = mae_sum / n_points
    mse = mse_sum / n_points
    return mae, mse


def compute_test_metrics_original_scale(model, loader, device, scaler):
    """
    Compute MAE and MSE on test set in ORIGINAL scale.
    """
    model.eval()
    mae_sum = 0.0
    mse_sum = 0.0
    n_points = 0

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

            pred = model(X)

            pred_np = pred.cpu().numpy()  # (B, C, H, W)
            Y_np = Y.cpu().numpy()        # (B, C, H, W)

            B, C, H, W = pred_np.shape

            # flatten per horizon (all channels are scaled the same here)
            pred_flat_norm = pred_np.reshape(B * C * H * W, 1)
            Y_flat_norm = Y_np.reshape(B * C * H * W, 1)

            pred_flat = scaler.inverse_transform(pred_flat_norm)
            Y_flat = scaler.inverse_transform(Y_flat_norm)

            diff = pred_flat - Y_flat
            mae_sum += np.abs(diff).sum()
            mse_sum += (diff ** 2).sum()
            n_points += diff.size

    mae = mae_sum / n_points
    mse = mse_sum / n_points
    return mae, mse


def main():
    device = cfg.device
    if not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")

    print("Preparing dataloaders (will load existing dataset)...")
    train_loader, val_loader, test_loader, scaler, train_idx, val_idx, test_idx = prepare_dataloaders()


    # 🔹 For reconstruction / visualization we need deterministic order:
    # train_eval_loader = DataLoader(train_loader.dataset,
    #                     batch_size=cfg.batch_size,
    #                     shuffle=False,)
    #
    # val_eval_loader = DataLoader(val_loader.dataset,
    #                     batch_size=cfg.batch_size,
    #                     shuffle=False,)
    #
    # test_eval_loader = DataLoader(test_loader.dataset,
    #                     batch_size=cfg.batch_size,
    #                     shuffle=False,)

    n_train = len(train_idx)
    n_val = len(val_idx)
    n_test = len(test_idx)

    print(f"Windows -> Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"Batches -> Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    model = build_model_from_cfg(cfg).to(device)

    # model = CNNGRUForecaster(
    #     in_channels=1,
    #     feature_dim=256,
    #     gru_hidden_dim=256,
    #     gru_layers=1,
    #     horizon_size=(cfg.H_out, cfg.W_out)
    # ).to(device)
    # model = UNet(
    #     in_channels=cfg.in_channels,
    #     out_channels=1,
    #     out_size=(cfg.H_out, cfg.W_out),  # enforce 15x24 output
    # ).to(device)
    print(f"Loading best model from {cfg.model_ckpt} ...")
    ckpt = torch.load(cfg.model_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("Computing test MAE and MSE (normalized scale)...")
    mae, mse = compute_test_metrics(model, test_loader, device)
    print(f"\nTest MAE (normalized): {mae:.6f}")
    print(f"Test MSE (normalized): {mse:.6f}")


    # print("\nVisualizing some sample images...")
    # visualize_sample_images(model, test_loader, device, scaler, num_samples=20)
    # visualize_sample_images(
    #     model,
    #     test_loader,
    #     device,
    #     scaler,
    #     num_samples=10,
    #     selection="random",  # or "worst" / "random"
    #     test_idx=test_idx,
    # )
    #
    print("\nVisualizing TEST prediction windows in seconds...")
    visualize_test_windows_sec_original(
        model, test_loader, test_idx, device, scaler,
        seq_len=5, num_windows=5
    )

    # visualize_test_windows_sec_normalized(
    #     model, test_loader, test_idx, device,
    #     seq_len=5, num_windows=5
    # )
    # visualize_test_windows_sec(
    #     model,
    #     test_loader=test_loader,
    #     test_idx=test_idx,
    #     device=device,
    #     scaler=scaler,
    #     seq_len=cfg.context_len,  # your sequence length
    #     num_windows=3,  # how many windows to plot
    # )

    print("\nVisualizing traffic of the whole dataset (normalized or original)...")
    # visualize_all_test_traffic_normalized(model, test_loader, test_idx, device)
    # visualize_all_test_traffic(model, test_loader, test_idx, device, scaler)

    # print("\nVisualizing traffic over WHOLE dataset (normalized)...")
    # visualize_full_dataset_traffic(model, train_eval_loader, val_eval_loader, test_loader, train_idx, val_idx, test_idx, device, scaler)
    # visualize_full_dataset_traffic_normalized(model, train_loader, val_loader, test_loader, train_idx, val_idx, test_idx, device)


if __name__ == "__main__":
    main()
