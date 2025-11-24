# train.py

import os
import torch
import torch.nn as nn

from config import cfg
from dataset import prepare_dataloaders
from models.unet2d import UNet

from models import build_model_from_cfg


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for X_seq, Y in loader:
        X_seq = X_seq.to(device) # (B, K, 1, H_in, W_in)
        Y = Y.to(device)

        pred = model(X_seq)
        loss = criterion(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)

            pred = model(X)
            loss = criterion(pred, Y)
            total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = cfg.device
    if not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")

    print("Preparing dataloaders (this will generate dataset if missing)...")
    # train_loader, val_loader, _, scaler, train_idx, val_idx, test_idx = prepare_dataloaders()
    train_loader, val_loader, test_loader, scaler, train_idx, val_idx, test_idx = prepare_dataloaders()

    n_train = len(train_idx)
    n_val = len(val_idx)
    print(f"Train windows: {n_train}, Val windows: {n_val}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # model = UNet(
    #     in_channels=cfg.in_channels,
    #     out_channels=1,
    #     out_size=(cfg.H_out, cfg.W_out),  # enforce 15x24 output
    # ).to(device)
    # model = CNNGRUForecaster(
    #     in_channels=1,
    #     feature_dim=256,
    #     gru_hidden_dim=256,
    #     gru_layers=1,
    #     horizon_size=(cfg.H_out, cfg.W_out)
    # ).to(device)
    model = build_model_from_cfg(cfg).to(device)
    print(model.__class__.__name__)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        # test_loss = evaluate(model, test_loader, criterion, device)

        print(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            # f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Test Loss: {test_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                cfg.model_ckpt,
            )
            print(f"  -> Saved best model to {cfg.model_ckpt}")


if __name__ == "__main__":
    main()
