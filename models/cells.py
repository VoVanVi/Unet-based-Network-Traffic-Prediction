# models/cell.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1) CNN ENCODER FOR 2D WINDOWS
# =========================================================
class CNNEncoder2D(nn.Module):
    """
    Encode a 2D window (C x H x W) into a feature vector.
    Can be reused for CNN-GRU, CNN-LSTM, etc.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_dim: int = 256,
        hidden_channels=(32, 64, 128),
    ):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in hidden_channels:
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  # H,W -> H/2,W/2
            c_in = c_out

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # (B, C, 1, 1)
        self.fc = nn.Linear(c_in, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        return: (B, feature_dim)
        """
        h = self.conv(x)
        h = self.pool(h)           # (B, C_last, 1, 1)
        h = h.view(h.size(0), -1)  # (B, C_last)
        out = self.fc(h)           # (B, feature_dim)
        return out


# =========================================================
# 2) GRU / LSTM DECODERS OVER FEATURES
# =========================================================
class GRUDecoder(nn.Module):
    """
    GRU over a sequence of feature vectors, then MLP to output_dim.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, seq_feat: torch.Tensor) -> torch.Tensor:
        """
        seq_feat: (B, K, input_dim)
        return: (B, output_dim)
        """
        out, _ = self.gru(seq_feat)    # (B, K, hidden_dim)
        h_last = out[:, -1, :]         # (B, hidden_dim)
        y = self.fc_out(h_last)        # (B, output_dim)
        return y


class LSTMDecoder(nn.Module):
    """
    LSTM over a sequence of feature vectors, then MLP to output_dim.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, seq_feat: torch.Tensor) -> torch.Tensor:
        """
        seq_feat: (B, K, input_dim)
        return: (B, output_dim)
        """
        out, _ = self.lstm(seq_feat)   # (B, K, hidden_dim)
        h_last = out[:, -1, :]         # (B, hidden_dim)
        y = self.fc_out(h_last)        # (B, output_dim)
        return y


# =========================================================
# 3) TCN (Temporal Convolutional Network) BLOCKS
# =========================================================
class Chomp1d(nn.Module):
    """
    Remove extra padding on the right to keep causality.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    A basic TCN block: dilated conv -> ReLU -> Dropout -> residual.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    A stack of TemporalBlock layers.
    Input/Output: (B, C, T)
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2**i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation
            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size,
                    stride=1,
                    dilation=dilation,
                    padding=padding,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T)
        return: (B, C_out, T)
        """
        return self.network(x)


# =========================================================
# 4) READY-MADE FORECASTER MODELS USING THESE CELLS
# =========================================================
class CNNGRUForecaster(nn.Module):
    """
    CNN encoder over each window + GRU over window sequence.

    Expected input:
        x_seq:
          - (B, K, C, H, W)  sequence of K windows
          - or (B, C, H, W)  single window (K=1)
    Output:
        (B, 1, H_out, W_out)
    """

    def __init__(
        self,
        encoder: CNNEncoder2D,
        decoder: GRUDecoder,
        H_out: int,
        W_out: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.H_out = H_out
        self.W_out = W_out

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # Accept both 4D and 5D
        if x_seq.dim() == 4:
            # (B, C, H, W) -> (B, 1, C, H, W)
            x_seq = x_seq.unsqueeze(1)

        B, K, C, H, W = x_seq.shape

        # Encode each window
        x_flat = x_seq.view(B * K, C, H, W)     # (B*K, C, H, W)
        feat = self.encoder(x_flat)             # (B*K, F)
        Fdim = feat.size(-1)
        feat_seq = feat.view(B, K, Fdim)        # (B, K, F)

        # GRU decoding
        y_flat = self.decoder(feat_seq)         # (B, H_out*W_out)
        y = y_flat.view(B, 1, self.H_out, self.W_out)
        return y


class CNNLSTMForecaster(nn.Module):
    """
    Same as CNNGRUForecaster but using LSTMDecoder.
    """

    def __init__(
        self,
        encoder: CNNEncoder2D,
        decoder: LSTMDecoder,
        H_out: int,
        W_out: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.H_out = H_out
        self.W_out = W_out

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() == 4:
            x_seq = x_seq.unsqueeze(1)
        B, K, C, H, W = x_seq.shape

        x_flat = x_seq.view(B * K, C, H, W)
        feat = self.encoder(x_flat)
        Fdim = feat.size(-1)
        feat_seq = feat.view(B, K, Fdim)

        y_flat = self.decoder(feat_seq)
        y = y_flat.view(B, 1, self.H_out, self.W_out)
        return y


class TCNForecaster1D(nn.Module):
    """
    Pure 1D temporal forecaster:
      - Input: (B, 1, T_in)
      - TCN -> last time step -> MLP to T_out
      - Output reshaped to (B, 1, H_out, W_out)
    """

    def __init__(
        self,
        T_in: int,
        H_out: int,
        W_out: int,
        tcn_channels=(64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.T_in = T_in
        self.H_out = H_out
        self.W_out = W_out
        self.tcn = TemporalConvNet(
            num_inputs=1,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        last_ch = tcn_channels[-1]
        self.fc = nn.Linear(last_ch, H_out * W_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, T_in)
        """
        out = self.tcn(x)          # (B, C_last, T_in)
        h_last = out[:, :, -1]     # (B, C_last)
        y_flat = self.fc(h_last)   # (B, H_out*W_out)
        y = y_flat.view(-1, 1, self.H_out, self.W_out)
        return y


# # =========================================================
# # 5) SIMPLE BUILDERS TO CALL FROM TRAIN/TEST
# # =========================================================
# def build_cnn_gru_forecaster_from_cfg(cfg) -> CNNGRUForecaster:
#     """
#     Convenience: build CNN+GRU forecaster from cfg.
#     Requires:
#         cfg.in_channels
#         cfg.cnn_feature_dim
#         cfg.gru_hidden_dim
#         cfg.gru_layers
#         cfg.H_out, cfg.W_out
#     """
#     encoder = CNNEncoder2D(
#         in_channels=cfg.in_channels,
#         feature_dim=cfg.cnn_feature_dim,
#     )
#     decoder = GRUDecoder(
#         input_dim=cfg.cnn_feature_dim,
#         hidden_dim=cfg.gru_hidden_dim,
#         num_layers=cfg.gru_layers,
#         output_dim=cfg.H_out * cfg.W_out,
#     )
#     return CNNGRUForecaster(
#         encoder=encoder,
#         decoder=decoder,
#         H_out=cfg.H_out,
#         W_out=cfg.W_out,
#     )
#
#
# def build_cnn_lstm_forecaster_from_cfg(cfg) -> CNNLSTMForecaster:
#     encoder = CNNEncoder2D(
#         in_channels=cfg.in_channels,
#         feature_dim=cfg.cnn_feature_dim,
#     )
#     decoder = LSTMDecoder(
#         input_dim=cfg.cnn_feature_dim,
#         hidden_dim=cfg.lstm_hidden_dim,
#         num_layers=cfg.lstm_layers,
#         output_dim=cfg.H_out * cfg.W_out,
#     )
#     return CNNLSTMForecaster(
#         encoder=encoder,
#         decoder=decoder,
#         H_out=cfg.H_out,
#         W_out=cfg.W_out,
#     )
#
#
# def build_tcn_forecaster_from_cfg(cfg) -> TCNForecaster1D:
#     """
#     For a pure 1D TCN forecaster.
#     Requires:
#         cfg.H_in, cfg.W_in       -> T_in = H_in * W_in
#         cfg.H_out, cfg.W_out
#     """
#     T_in = cfg.H_in * cfg.W_in
#     return TCNForecaster1D(
#         T_in=T_in,
#         H_out=cfg.H_out,
#         W_out=cfg.W_out,
#         tcn_channels=getattr(cfg, "tcn_channels", (64, 64, 64)),
#         kernel_size=getattr(cfg, "tcn_kernel_size", 3),
#         dropout=getattr(cfg, "tcn_dropout", 0.2),
#     )


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

# def build_unet_forecaster_from_cfg(cfg) -> TCNForecaster1D:
#     """
#     For a pure 1D TCN forecaster.
#     Requires:
#         cfg.H_in, cfg.W_in       -> T_in = H_in * W_in
#         cfg.H_out, cfg.W_out
#     """
#     T_in = cfg.H_in * cfg.W_in
#     return TCNForecaster1D(
#         T_in=T_in,
#         H_out=cfg.H_out,
#         W_out=cfg.W_out,
#         tcn_channels=getattr(cfg, "tcn_channels", (64, 64, 64)),
#         kernel_size=getattr(cfg, "tcn_kernel_size", 3),
#         dropout=getattr(cfg, "tcn_dropout", 0.2),
#     )