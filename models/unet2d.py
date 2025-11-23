# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=(32, 64, 128, 256),
        out_size=None,  # NEW: (H_out, W_out) or None to keep input size
    ):
        super().__init__()

        self.out_size = out_size  # (H_out, W_out) or None

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down path
        ch = in_channels
        for f in features:
            block = nn.Sequential(
                nn.Conv2d(ch, f, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(f, f, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.downs.append(block)
            ch = f

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[-1], features[-1] * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[-1] * 2, features[-1] * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Up path
        for f in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
            )
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(f * 2, f, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(f, f, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )

        # Final conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Reverse skip list
        skip_connections = skip_connections[::-1]

        # Upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # ConvTranspose
            skip = skip_connections[idx // 2]

            # Align spatial sizes before concat (because odd sizes)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )

            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)  # Conv + Conv

        x = self.final_conv(x)  # (B, out_channels, H_in', W_in')

        # 🔸 NEW: enforce desired output size if provided
        if self.out_size is not None:
            H_out, W_out = self.out_size
            if x.shape[2] != H_out or x.shape[3] != W_out:
                x = F.interpolate(
                    x,
                    size=(H_out, W_out),
                    mode="bilinear",
                    align_corners=False,
                )

        return x

def build_unet_from_cfg(cfg) -> UNet:
    """
    Build a UNet model from cfg.

    Expects in cfg:
        cfg.in_channels          # input channels, e.g., 1 for gray2d, 3 for rgb2d
        cfg.out_channels         # usually 1 for single traffic metric
        cfg.H_out, cfg.W_out     # desired output image size
        cfg.unet_features        # e.g., (32, 64, 128, 256)
    """
    features = getattr(cfg, "unet_features", (32, 64, 128, 256))
    out_size = (cfg.H_out, cfg.W_out) if getattr(cfg, "H_out", None) is not None else None

    return UNet(
        in_channels=cfg.in_channels,
        out_channels=getattr(cfg, "out_channels", 1),
        features=features,
        out_size=out_size,
    )
