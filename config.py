# config.py

import os


class Config:
    # ---------- Data paths ----------
    raw_csv_path = os.path.join("data/raw", "Pangyo_traffic_volume.csv")
    value_col = "All packets"

    # ---------- Image / sequence settings ----------
    H_in = 63
    W_in = 24
    H_out = 15
    W_out = 24

    context_len = 5  # or 5, etc. Number of past windows for CNN-GRU

    overlap_rows = 3          # 24 rows overlapped -> 24*24 = 576 steps
    stride = 1                 # 10 sec shift per image

    image_mode = "gray2d"      # or "rgb2d" or "spectrogram"

    # in_channels = 1            # gray2d -> 1, rgb2d -> 3, ...
    if image_mode in ["gray2d", "spectrogram"]:
        in_channels = 1
    elif image_mode == "rgb2d":
        in_channels = 3

    # ---------- Sampling ----------
    sample_interval_sec = 10   # 10s per raw sample

    # ---------- Splits ----------
    val_ratio = 0.1
    test_ratio = 0.2
    random_seed = 42

    # ---------- Model ----------
    model_type = "unet"  # or "cnn_gru", "cnn_lstm", "tcn", "unet", "coregan"
    in_channels = 1  # 1 for gray2d/spectrogram, 3 for rgb2d
    out_channels = 1
    cnn_feature_dim = 256
    gru_hidden_dim = 256
    gru_layers = 1
    lstm_hidden_dim = 256
    lstm_layers = 1
    tcn_channels = (64, 64, 64)
    tcn_kernel_size = 3
    tcn_dropout = 0.2
    unet_features = (32, 64, 128, 256)

    # ---------- Training ----------
    batch_size = 16
    num_epochs = 100
    lr = 1e-4

    # ---------- Device ----------
    device = "cuda" # "cuda" or "cpu"

    # ---------- Directories ----------
    checkpoint_dir = "checkpoints"
    results_dir = "results"
    visualized_dir = os.path.join(results_dir, "visualize")
    os.makedirs(visualized_dir, exist_ok=True)
    processed_dir = os.path.join("data", "processed")

    # dummy placeholder, we’ll override after cfg is created
    model_ckpt = None
    scaler_path = None

cfg = Config()


# ---- add this block ----
# A unique tag for this configuration (input/output size + stride)
cfg.image_tag = (
    f"in{cfg.H_in}x{cfg.W_in}_"
    f"out{cfg.H_out}x{cfg.W_out}_"
    f"overlap{cfg.overlap_rows}x{cfg.W_in}"
)

# Use the tag in all important artifact filenames
os.makedirs(cfg.checkpoint_dir, exist_ok=True)
cfg.model_ckpt = os.path.join(cfg.checkpoint_dir, f"{cfg.model_type}_best_{cfg.image_tag}.pt")
cfg.scaler_path = os.path.join(cfg.checkpoint_dir, f"scaler_{cfg.model_type}_{cfg.image_tag}.pkl")