"""
PyTorch reimplementation of the original TensorFlow CoreGAN model.

It provides:
- CoreGANGenerator: ConvLSTM-style generator that consumes a sequence of
  traffic images and predicts the next frame.
- CoreGANDiscriminator: CNN discriminator that returns both the validity
  score and an intermediate feature vector.
- CoreGANTrainer: lightweight GAN + feature-matching training loop so the
  model can be trained/tested similarly to the other PyTorch models in
  this repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class ConvLSTMCell(nn.Module):
    """A single ConvLSTM cell.

    This is intentionally minimal to mirror the TensorFlow ConvLSTM2D layers
    used in the original COREGAN implementation while staying dependency-free.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int,
        padding: Union[int, str],
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMBlock(nn.Module):
    """Stack the ConvLSTM cell over the time dimension."""

    def __init__(
        self, input_dim: int, hidden_dim: int, kernel_size: int, padding: Union[int, str]
    ):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, padding)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a sequence.

        Args:
            x: Tensor of shape (B, T, C, H, W).

        Returns:
            all_hidden: hidden states for every timestep (B, T, hidden_dim, H, W)
            last_hidden: final hidden state (B, hidden_dim, H, W)
        """

        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.cell.hidden_dim, H, W, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)

        outputs: List[torch.Tensor] = []
        for t in range(T):
            h, c = self.cell(x[:, t], (h, c))
            outputs.append(h)

        all_hidden = torch.stack(outputs, dim=1)
        return all_hidden, h


# ---------------------------------------------------------------------------
# Generator & Discriminator
# ---------------------------------------------------------------------------
class CoreGANGenerator(nn.Module):
    """ConvLSTM-based generator.

    Input : (B, T, C, H, W)
    Output: (B, out_channels, H, W)
    """

    def __init__(self, time_steps: int, in_channels: int, H: int, W: int, out_channels: int = 1):
        super().__init__()
        self.time_steps = time_steps
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Mirror the original architecture: four ConvLSTM layers then a 2D conv
        self.lstm1 = ConvLSTMBlock(in_channels, 40, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm3d(40)
        self.lstm2 = ConvLSTMBlock(40, 40, kernel_size=2, padding="same")
        self.bn2 = nn.BatchNorm3d(40)
        self.lstm3 = ConvLSTMBlock(40, 40, kernel_size=2, padding="same")
        self.bn3 = nn.BatchNorm3d(40)
        self.lstm4 = ConvLSTMBlock(40, 40, kernel_size=2, padding="same")
        self.bn4 = nn.BatchNorm2d(40)
        self.final_conv = nn.Conv2d(40, out_channels, kernel_size=3, padding=1)

        # Save sizes for reference / debugging
        self.H = H
        self.W = W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        seq1, _ = self.lstm1(x)
        seq1 = self.bn1(seq1.transpose(1, 2)).transpose(1, 2)  # BN over channels

        seq2, _ = self.lstm2(seq1)
        seq2 = self.bn2(seq2.transpose(1, 2)).transpose(1, 2)

        seq3, _ = self.lstm3(seq2)
        seq3 = self.bn3(seq3.transpose(1, 2)).transpose(1, 2)

        _, h4 = self.lstm4(seq3)
        h4 = self.bn4(h4)

        out = torch.sigmoid(self.final_conv(h4))
        return out

    # ------------------------------------------------------------------
    # Legacy checkpoint support
    # ------------------------------------------------------------------
    @staticmethod
    def convert_legacy_state_dict(state_dict: dict) -> Tuple[dict, bool]:
        """Convert older CoreGAN checkpoints to the current key layout.

        Earlier iterations of the PyTorch port saved the ConvLSTM stack as
        nested "convlstm_stack" modules (e.g., "convlstm_stack.cells.0.conv"
        and "convlstm_stack.norms.0"). The current implementation exposes the
        layers individually as ``lstm{1-4}``, ``bn{1-4}``, and ``final_conv``.

        This helper remaps the old keys when detected so users can load their
        previously trained checkpoints without retraining.

        Args:
            state_dict: Raw state_dict loaded from a checkpoint.

        Returns:
            (converted_state_dict, converted) where ``converted`` indicates
            whether any legacy keys were remapped.
        """

        def _remap_lstm(key: str) -> Optional[str]:
            prefix = "convlstm_stack.cells."
            if not key.startswith(prefix):
                return None
            # convlstm_stack.cells.{idx}.conv.{param}
            parts = key.split(".")
            if len(parts) < 5:
                return None
            idx = int(parts[2])  # 0-based
            param = ".".join(parts[4:])
            return f"lstm{idx + 1}.cell.conv.{param}"

        def _remap_norm(key: str) -> Optional[str]:
            prefix = "convlstm_stack.norms."
            if not key.startswith(prefix):
                return None
            parts = key.split(".")
            if len(parts) < 4:
                return None
            idx = int(parts[2])  # 0-based
            param = ".".join(parts[3:])
            return f"bn{idx + 1}.{param}"

        converted = False
        new_state = {}
        for key, val in state_dict.items():
            if key.startswith("convlstm_stack.cells."):
                new_key = _remap_lstm(key)
                if new_key:
                    converted = True
                    new_state[new_key] = val
                    continue
            if key.startswith("convlstm_stack.norms."):
                new_key = _remap_norm(key)
                if new_key:
                    converted = True
                    new_state[new_key] = val
                    continue
            if key.startswith("out_conv."):
                converted = True
                new_state[key.replace("out_conv", "final_conv", 1)] = val
                continue

            # Unrecognized key; keep as-is so load_state_dict can warn if needed.
            new_state[key] = val

        return new_state, converted

class CoreGANDiscriminator(nn.Module):
    """CNN discriminator for CoRe-GAN.

    - 3 Conv blocks with ReLU + MaxPool.
    - Adaptive pooling before the linear layers so we are not hard-coded
      to 15x24 input; by default we force a 3x5 feature map, which
      matches the original 15x24 setup but also works for other sizes.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        feat_dim: int = 128,
        pooled_hw: Tuple[int, int] = (3, 5),
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=2, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        # third conv, no extra pooling (paper: 3 conv + 2 pools)
        self.conv3 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        # adapt spatial size to (H', W') = pooled_hw (e.g., 3x5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(pooled_hw)

        self.flatten = nn.Flatten()
        self.feat = nn.Linear(base_channels * pooled_hw[0] * pooled_hw[1], feat_dim)
        self.valid = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        h = F.relu(self.conv1(x))
        h = self.pool1(h)

        h = F.relu(self.conv2(h))
        h = self.pool2(h)

        h = F.relu(self.conv3(h))
        h = self.adaptive_pool(h)

        flat = self.flatten(h)
        feat = F.relu(self.feat(flat))
        validity = torch.sigmoid(self.valid(feat))
        return validity, feat

# class CoreGANDiscriminator(nn.Module):
#     def __init__(self, in_channels: int = 1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=2)
#         self.pool1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=2)
#         self.pool2 = nn.MaxPool2d(2)
#         self.flatten = nn.Flatten()
#         self.feat = nn.Linear(64 * 3 * 5, 128)  # assume 15x24 input -> after pooling -> 3x5
#         self.valid = nn.Linear(128, 1)
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         h = F.relu(self.conv1(x))
#         h = self.pool1(h)
#         h = F.relu(self.conv2(h))
#         h = self.pool2(h)
#         flat = self.flatten(h)
#         feat = F.relu(self.feat(flat))
#         validity = torch.sigmoid(self.valid(feat))
#         return validity, feat


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------
@dataclass
class CoreGANConfig:
    epochs: int = 200
    batch_size: int = 128
    lr: float = 1e-4
    feature_matching_weight: float = 1.0
    recon_weight: float = 1.0
    adv_weight: float = 0.0
    device: str = "cuda:0"
    save_dir: str = "./results_coregan"
    log_interval: int = 1  # how often to print epoch metrics
    early_stopping_patience: int = 15  # 0 disables early stopping
    early_stopping_min_delta: float = 0.0

    @classmethod
    def from_global_cfg(cls, global_cfg: object) -> "CoreGANConfig":
        """Build a CoreGANConfig using values defined in config.py when available."""

        defaults = cls()
        return cls(
            epochs=getattr(global_cfg, "coregan_epochs", getattr(global_cfg, "num_epochs", defaults.epochs)),
            batch_size=getattr(
                global_cfg, "coregan_batch_size", getattr(global_cfg, "batch_size", defaults.batch_size)
            ),
            lr=getattr(global_cfg, "coregan_lr", getattr(global_cfg, "lr", defaults.lr)),
            feature_matching_weight=getattr(
                global_cfg, "coregan_feature_matching_weight", defaults.feature_matching_weight
            ),
            recon_weight=getattr(global_cfg, "coregan_recon_weight", defaults.recon_weight),
            adv_weight=getattr(global_cfg, "coregan_adv_weight", defaults.adv_weight),
            device=getattr(global_cfg, "device", defaults.device),
            save_dir=getattr(global_cfg, "coregan_save_dir", defaults.save_dir),
            log_interval=getattr(global_cfg, "coregan_log_interval", defaults.log_interval),
            early_stopping_patience=getattr(
                global_cfg, "coregan_early_stopping_patience", defaults.early_stopping_patience
            ),
            early_stopping_min_delta=getattr(
                global_cfg, "coregan_early_stopping_min_delta", defaults.early_stopping_min_delta
            ),
        )

class FeatureMatchingL2Loss(nn.Module):
    """
    L_FM = E[ || f_real - f_fake ||_2 ]
    where f_* are feature vectors from the discriminator.
    """
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, fake_feats: torch.Tensor, real_feats: torch.Tensor) -> torch.Tensor:
        # fake_feats, real_feats: (B, F)
        diff = fake_feats - real_feats
        # per-sample L2 norm, then mean over batch
        return torch.norm(diff, p=2, dim=self.dim).mean()



class CoreGANTrainer:
    """Minimal GAN training loop for CoreGAN using PyTorch."""

    def __init__(
        self,
        generator: CoreGANGenerator,
        discriminator: CoreGANDiscriminator,
        cfg: Optional[CoreGANConfig] = None,
    ):
        self.cfg = cfg or CoreGANConfig()
        self.device = torch.device(self.cfg.device)

        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

        self.opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=self.cfg.lr)
        self.opt_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.cfg.lr)

        self.adv_criterion = nn.BCELoss()
        self.recon_criterion = nn.MSELoss()
        self.fm_criterion = FeatureMatchingL2Loss(dim=1)
        # self.fm_criterion = nn.MSELoss()

        os.makedirs(self.cfg.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cfg.save_dir, "csv"), exist_ok=True)
        os.makedirs(os.path.join(self.cfg.save_dir, "pred"), exist_ok=True)

    def _prepare_labels(self, batch_size: int, device: torch.device):
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)
        return valid, fake

    def _maybe_save_best(
        self,
        checkpoint_path: Optional[str],
        monitor_value: float,
        best_value: float,
        metrics: Dict[str, List[float]],
        monitor_label: str,
    ) -> Tuple[float, bool]:
        """Save best-performing generator/discriminator like train.py."""

        improved = monitor_value < (best_value - self.cfg.early_stopping_min_delta)
        if not improved:
            return best_value, False

        if checkpoint_path is None:
            return monitor_value, True

        best_state_gen = {
            k: v.detach().cpu().clone() for k, v in self.generator.state_dict().items()
        }
        best_state_disc = {
            k: v.detach().cpu().clone()
            for k, v in self.discriminator.state_dict().items()
        }
        payload = {
            "model_state_dict": best_state_gen,
            "generator_state_dict": best_state_gen,
            "discriminator_state_dict": best_state_disc,
            "metrics": metrics,
            f"best_{monitor_label}": monitor_value,
        }
        torch.save(payload, checkpoint_path)
        print(f"  -> Saved best model to {checkpoint_path}")
        return monitor_value, True

    def train(
        self,
        train_loader,
        val_loader=None,
        test_loader=None,
        checkpoint_path: Optional[str] = None,
    ):
        g_losses, d_losses, fm_losses, recon_losses = [], [], [], []
        best_monitor = float("inf")
        stale_epochs = 0

        patience = self.cfg.early_stopping_patience

        for epoch in range(self.cfg.epochs):
            epoch_g, epoch_d, epoch_fm, epoch_recon = [], [], [], []
            for x_seq, y in train_loader:
                x_seq = x_seq.to(self.device)
                y = y.to(self.device)
                if y.ndim == 3:
                    y = y.unsqueeze(1)

                batch_size = x_seq.size(0)
                valid, fake = self._prepare_labels(batch_size, self.device)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.opt_d.zero_grad()
                with torch.no_grad():
                    fake_imgs = self.generator(x_seq)

                real_logits, _ = self.discriminator(y)
                fake_logits, _ = self.discriminator(fake_imgs.detach())

                d_loss_real = self.adv_criterion(real_logits, valid)
                d_loss_fake = self.adv_criterion(fake_logits, fake)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                d_loss.backward()
                self.opt_d.step()

                # -----------------
                #  Train Generator
                # -----------------
                self.opt_g.zero_grad()

                # forward pass through generator
                gen_imgs = self.generator(x_seq)

                # discriminator outputs used ONLY for feature matching (not BCE)
                fake_logits, fake_feats = self.discriminator(gen_imgs)
                with torch.no_grad():
                    _, real_feats = self.discriminator(y)

                # fact forcing = pixel-wise reconstruction (MSE) between gen_imgs and y
                recon_loss = self.recon_criterion(gen_imgs, y)  # L_FF in the paper

                # feature matching = L2 between intermediate feature representations
                fm_loss = self.fm_criterion(fake_feats, real_feats)  # L_FM in the paper

                # CoRe-GAN generator loss: feature matching + fact forcing
                g_loss = (
                        self.cfg.recon_weight * recon_loss
                        + self.cfg.feature_matching_weight * fm_loss
                )

                # (optional) if you really want to experiment with adversarial BCE,
                # you can keep this term but it's OFF by default because adv_weight=0.0.
                if self.cfg.adv_weight != 0.0:
                    adv_loss = self.adv_criterion(fake_logits, valid)
                    g_loss = g_loss + self.cfg.adv_weight * adv_loss

                g_loss.backward()
                self.opt_g.step()

                epoch_g.append(g_loss.item())
                epoch_d.append(d_loss.item())
                epoch_fm.append(fm_loss.item())
                epoch_recon.append(recon_loss.item())
                # epoch_adv.append(adv_loss.item())

            g_losses.append(sum(epoch_g) / len(epoch_g))
            d_losses.append(sum(epoch_d) / len(epoch_d))
            fm_losses.append(sum(epoch_fm) / len(epoch_fm))
            recon_losses. append(sum(epoch_recon) / len(epoch_recon))
            # adv_losses.append(sum(epoch_adv) / len(epoch_adv))

            val_mse = None
            if val_loader is not None:
                val_mse = self.evaluate(val_loader)

            metrics = {
                "G_Loss": g_losses,
                "D_Loss": d_losses,
                "FM_Loss": fm_losses,
                "FF_Loss": recon_losses,
                # "ADV_Loss": adv_losses,
            }
            monitor_label = "val_mse" if val_mse is not None else "g_loss"
            monitor_value = val_mse if val_mse is not None else g_losses[-1]
            metrics[monitor_label.upper()] = (
                metrics.get(monitor_label.upper(), []) + [monitor_value]
            )
            best_monitor, improved = self._maybe_save_best(
                checkpoint_path, monitor_value, best_monitor, metrics, monitor_label
            )

            if improved:
                stale_epochs = 0
            else:
                stale_epochs += 1

            if patience > 0 and stale_epochs >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1}: "
                    f"no improvement in {patience} epochs."
                )
                break

            if (epoch + 1) % self.cfg.log_interval == 0:
                log_msg = (
                    f"Epoch {epoch+1}/{self.cfg.epochs} | "
                    f"D_loss={d_losses[-1]:.4f} | "
                    f"G_loss={g_losses[-1]:.4f} | "
                )
                if val_mse is not None:
                    log_msg += f"Val_MSE={val_mse:.4f} | "
                log_msg += (
                    f"FM={fm_losses[-1]:.4f} | " f"FF={recon_losses[-1]:.4f}"
                )
                print(log_msg)

        metrics = {
            "G_Loss": g_losses,
            "D_Loss": d_losses,
            "FM_Loss": fm_losses,
            "FF_Loss": recon_losses,
        }

        test_mse = None
        if test_loader is not None:
            test_mse = self.evaluate(test_loader)
        return metrics, test_mse

    @torch.no_grad()
    def evaluate(self, data_loader):
        self.generator.eval()
        mse_list: List[float] = []
        for x_seq, y in data_loader:
            x_seq = x_seq.to(self.device)
            y = y.to(self.device)
            if y.ndim == 3:
                y = y.unsqueeze(1)
            pred = self.generator(x_seq)
            mse = F.mse_loss(pred, y, reduction="mean").item()
            mse_list.append(mse)
        self.generator.train()
        return sum(mse_list) / len(mse_list)


__all__ = [
    "ConvLSTMCell",
    "ConvLSTMBlock",
    "CoreGANGenerator",
    "CoreGANDiscriminator",
    "CoreGANConfig",
    "CoreGANTrainer",
]