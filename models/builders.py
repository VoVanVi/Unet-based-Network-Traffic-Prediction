# build_cnn_gru..., build_unet..., build_model_from_cfg

# models/builders.py

import torch.nn as nn

from .cells import (
    CNNEncoder2D,
    GRUDecoder,
    LSTMDecoder,
    TemporalConvNet,
    TCNForecaster1D,
    CNNGRUForecaster,
    CNNLSTMForecaster,
    UNet,
)

from .coregan_pytorch import CoreGANGenerator


# =========================================================
# 1) Individual builders
# =========================================================
def build_cnn_gru_forecaster_from_cfg(cfg) -> CNNGRUForecaster:
    """
    Convenience: build CNN+GRU forecaster from cfg.
    Requires:
        cfg.in_channels
        cfg.cnn_feature_dim
        cfg.gru_hidden_dim
        cfg.gru_layers
        cfg.H_out, cfg.W_out
    """
    encoder = CNNEncoder2D(
        in_channels=cfg.in_channels,
        feature_dim=cfg.cnn_feature_dim,
    )
    decoder = GRUDecoder(
        input_dim=cfg.cnn_feature_dim,
        hidden_dim=cfg.gru_hidden_dim,
        num_layers=cfg.gru_layers,
        output_dim=cfg.H_out * cfg.W_out,
    )
    return CNNGRUForecaster(
        encoder=encoder,
        decoder=decoder,
        H_out=cfg.H_out,
        W_out=cfg.W_out,
    )


def build_cnn_lstm_forecaster_from_cfg(cfg) -> CNNLSTMForecaster:
    encoder = CNNEncoder2D(
        in_channels=cfg.in_channels,
        feature_dim=cfg.cnn_feature_dim,
    )
    decoder = LSTMDecoder(
        input_dim=cfg.cnn_feature_dim,
        hidden_dim=cfg.lstm_hidden_dim,
        num_layers=cfg.lstm_layers,
        output_dim=cfg.H_out * cfg.W_out,
    )
    return CNNLSTMForecaster(
        encoder=encoder,
        decoder=decoder,
        H_out=cfg.H_out,
        W_out=cfg.W_out,
    )


def build_tcn_forecaster_from_cfg(cfg) -> TCNForecaster1D:
    T_in = cfg.H_in * cfg.W_in
    return TCNForecaster1D(
        T_in=T_in,
        H_out=cfg.H_out,
        W_out=cfg.W_out,
        tcn_channels=getattr(cfg, "tcn_channels", (64, 64, 64)),
        kernel_size=getattr(cfg, "tcn_kernel_size", 3),
        dropout=getattr(cfg, "tcn_dropout", 0.2),
    )


def build_unet_from_cfg(cfg) -> UNet:
    """
    Build a UNet model from cfg.

    Expects in cfg:
        cfg.in_channels      # 1 for gray2d/spectrogram, 3 for rgb2d
        cfg.out_channels     # usually 1
        cfg.H_out, cfg.W_out
        cfg.unet_features    # tuple of feature sizes
    """
    features = getattr(cfg, "unet_features", (32, 64, 128, 256))
    out_size = (cfg.H_out, cfg.W_out) if getattr(cfg, "H_out", None) is not None else None

    return UNet(
        in_channels=cfg.in_channels,
        out_channels=getattr(cfg, "out_channels", 1),
        features=features,
        out_size=out_size,
    )


# =========================================================
# 2) A single entry point: build_model_from_cfg
# =========================================================
def build_model_from_cfg(cfg) -> nn.Module:
    """
    Master factory function to build the right model based on cfg.model_type.

    cfg.model_type in {"unet", "cnn_gru", "cnn_lstm", "tcn"}
    """
    model_type = getattr(cfg, "model_type", "unet").lower()

    if model_type == "unet":
        return build_unet_from_cfg(cfg)
    elif model_type == "cnn_gru":
        return build_cnn_gru_forecaster_from_cfg(cfg)
    elif model_type == "cnn_lstm":
        return build_cnn_lstm_forecaster_from_cfg(cfg)
    elif model_type == "tcn":
        return build_tcn_forecaster_from_cfg(cfg)
    elif cfg.model_type == "coregan":
        # For test/inference we only need the generator, discriminator is used in training.
        return CoreGANGenerator(
            time_steps=cfg.context_len,
            in_channels=cfg.in_channels,
            H=cfg.H_out,  # CoreGAN assumes H_in == H_out, W_in == W_out
            W=cfg.W_out,
        )
    else:
        raise ValueError(f"Unknown cfg.model_type: {cfg.model_type}")