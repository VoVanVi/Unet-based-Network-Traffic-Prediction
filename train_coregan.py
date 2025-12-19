"""
Adversarial training entry point for the PyTorch CoreGAN implementation.

This script mirrors the dataset preparation used by the other models but trains
CoreGAN with the adversarial + feature-matching losses defined in
``models.coregan.CoreGANTrainer``.
"""

import os
import torch

from config import cfg
from dataset import prepare_dataloaders
from models.coregan_pytorch import (
    CoreGANConfig,
    CoreGANDiscriminator,
    CoreGANGenerator,
    CoreGANTrainer,
)


def main():
    device = cfg.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Preparing dataloaders (this will generate the dataset if missing)...")
    train_loader, val_loader, test_loader, scaler, train_idx, val_idx, test_idx = prepare_dataloaders()
    print(
        f"Windows -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}"
    )
    print(
        f"Batches -> Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}"
    )

    generator = CoreGANGenerator(
        time_steps=cfg.context_len,
        in_channels=cfg.in_channels,
        H=cfg.H_in,
        W=cfg.W_in,
        out_channels=cfg.out_channels,
    )
    discriminator = CoreGANDiscriminator(in_channels=cfg.out_channels)

    trainer_cfg = CoreGANConfig(
        epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=device,
        save_dir=os.path.join(cfg.results_dir, "coregan_gan"),
    )

    ckpt_path = cfg.model_ckpt
    trainer = CoreGANTrainer(generator, discriminator, trainer_cfg)
    metrics, test_mse = trainer.train(
        train_loader, test_loader, checkpoint_path=ckpt_path
    )

    print(f"Best CoreGAN checkpoint saved to {ckpt_path}")
    if test_mse is not None:
        print(f"Test MSE (normalized): {test_mse:.6f}")


if __name__ == "__main__":
    main()