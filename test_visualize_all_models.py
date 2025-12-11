import copy
import os
import numpy as np
import torch

from config import cfg
from models import build_model_from_cfg
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import joblib


# --------------------------
# Helper: dataset path name
# --------------------------
def _get_dataset_path(image_tag):
    os.makedirs(cfg.processed_dir, exist_ok=True)
    fname = f"dataset_{cfg.model_type}_{cfg.image_mode}_{image_tag}.npz"
    return os.path.join(cfg.processed_dir, fname)

# -------------------------------------
# Load dataset from disk
# -------------------------------------
def _load_dataset_arrays(image_tag):
    """
    Load dataset arrays and meta from .npz.
    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test,
        train_idx, val_idx, test_idx
    """
    dataset_path = _get_dataset_path(image_tag)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=False)
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_val   = data["X_val"]
    Y_val   = data["Y_val"]
    X_test  = data["X_test"]
    Y_test  = data["Y_test"]

    train_idx = data["train_idx"]
    val_idx   = data["val_idx"]
    test_idx  = data["test_idx"]

    # Optional: we could verify H_in/W_in/etc match cfg, but skipping for brevity

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, train_idx, val_idx, test_idx

# --------------------------
# Torch Dataset wrapper
# --------------------------
class TrafficImageDataset(Dataset):
    def __init__(self, X, Y):
        """
        X: numpy array of shape
           - (N, H, W)             single-channel images
           - (N, C, H, W)          multi-channel images
           - (N, K, C, H, W)       sequences of K windows (for CNN-GRU)
        Y: same but with H_out, W_out, usually:
           - (N, H_out, W_out)
           - (N, C_out, H_out, W_out)
        """
        X = np.asarray(X)
        Y = np.asarray(Y)

        # ---- X handling ----
        if X.ndim == 3:
            # (N, H, W) -> (N, 1, H, W)
            self.X = torch.from_numpy(X).unsqueeze(1)
        elif X.ndim == 4:
            # (N, C, H, W)
            self.X = torch.from_numpy(X)
        elif X.ndim == 5:
            # (N, K, C, H, W)
            self.X = torch.from_numpy(X)
        else:
            raise ValueError(f"Unexpected X shape: {X.shape}")

        # ---- Y handling ----
        if Y.ndim == 3:
            # (N, H_out, W_out) -> (N, 1, H_out, W_out)
            self.Y = torch.from_numpy(Y).unsqueeze(1)
        elif Y.ndim == 4:
            # (N, C_out, H_out, W_out)
            self.Y = torch.from_numpy(Y)
        else:
            raise ValueError(f"Unexpected Y shape: {Y.shape}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# -------------------------------------
# Public API: prepare_dataloaders()
# -------------------------------------
def prepare_dataloaders(image_tag):
    """
    Main entry point for train.py / test.py

    - If dataset file does NOT exist, generate and save it
    - Load dataset arrays from disk
    - Load scaler
    - Wrap in PyTorch DataLoaders

    Returns:
        train_loader, val_loader, test_loader, scaler,
        train_idx, val_idx, test_idx
    """
    dataset_path = _get_dataset_path(image_tag)

    print(f"[dataset] Using cached dataset: {dataset_path}")

    # Load arrays
    X_train, Y_train, X_val, Y_val, X_test, Y_test, train_idx, val_idx, test_idx = (
        _load_dataset_arrays(image_tag)
    )

    # Load scaler
    if not os.path.exists(cfg.scaler_path):
        raise FileNotFoundError(
            f"Scaler not found at {cfg.scaler_path}. "
            "Delete dataset .npz and regenerate if needed."
        )
    scaler = joblib.load(cfg.scaler_path)

    # Build DataLoaders
    train_ds = TrafficImageDataset(X_train, Y_train)
    val_ds   = TrafficImageDataset(X_val,   Y_val)
    test_ds  = TrafficImageDataset(X_test,  Y_test)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        scaler,
        train_idx,
        val_idx,
        test_idx,
    )

# -----------------------------
# Helper: run model on a loader
# -----------------------------
def _run_inference(model, loader, device):
    """
    Run model over entire loader and collect predictions + Ground Truth (normalized).
    Returns:
        preds: (N, C, H, W)
        gts:   (N, C, H, W)
    """
    model.eval()
    preds_list, gts_list = [], []

    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)
            pred = model(X)  # (B, C, H, W) or (B, 1, H, W)
            preds_list.append(pred.cpu().numpy())
            gts_list.append(Y.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    gts = np.concatenate(gts_list, axis=0)
    return preds, gts


def load_model_and_data_for_type(model_type: str, device):
    """
    Temporarily switch cfg.model_type to `model_type`,
    load its dataset (.npz), scaler, and checkpoint, then
    restore cfg.model_type.

    Assumes checkpoints are stored as:
        checkpoints/{model_type}_best_{cfg.image_tag}.pt
    and scalers as:
        checkpoints/scaler_{model_type}_{cfg.image_tag}.pkl
    """
    # Save original config state
    orig_type = cfg.model_type
    orig_ckpt = cfg.model_ckpt
    orig_scaler_path = cfg.scaler_path

    # Update cfg for this model_type
    cfg.model_type = model_type
    cfg.model_ckpt = os.path.join(
        cfg.checkpoint_dir,
        f"{model_type}_best_{cfg.image_tag}.pt"
    )
    cfg.scaler_path = os.path.join(
        cfg.checkpoint_dir,
        f"scaler_{model_type}_{cfg.image_tag}.pkl"
    )

    print(f"\n[LOAD] ==== model_type={model_type} ====")
    if model_type == 'unet':
        image_tag = "in27x24_out15x24_overlap12x24"
        cfg.model_ckpt = os.path.join(cfg.checkpoint_dir, f"{cfg.model_type}_best_{image_tag}.pt")
        cfg.scaler_path = os.path.join(cfg.checkpoint_dir, f"scaler_{cfg.model_type}_{image_tag}.pkl")
    else:
        image_tag = "in15x24_out15x24_overlap12x24"
        cfg.model_ckpt = os.path.join(cfg.checkpoint_dir, f"{cfg.model_type}_best_{image_tag}.pt")
        cfg.scaler_path = os.path.join(cfg.checkpoint_dir, f"scaler_{cfg.model_type}_{image_tag}.pkl")
    print(f"[LOAD] ckpt:   {cfg.model_ckpt}")
    print(f"[LOAD] scaler: {cfg.scaler_path}")

    # Load dataloaders & scaler for this model_type
    train_loader, val_loader, test_loader, scaler, train_idx, val_idx, test_idx = prepare_dataloaders(image_tag)

    # Build and load model
    model = build_model_from_cfg(cfg).to(device)

    ckpt = torch.load(cfg.model_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Restore original cfg
    cfg.model_type = orig_type
    cfg.model_ckpt = orig_ckpt
    cfg.scaler_path = orig_scaler_path

    return model, test_loader, scaler, test_idx


def reconstruct_series_original_scale(
    model,
    test_loader,
    test_idx,
    device,
    scaler,
):
    """
    Run inference on TEST set for a given model + loader,
    and reconstruct a SINGLE 1D predicted series and 1D GT series
    in ORIGINAL SCALE (using the same logic as your
    visualize_test_windows_sec_original, but without plotting).

    Returns:
        compressed_pred_orig : 1D np.ndarray, shape (L,)
        compressed_gt_orig   : 1D np.ndarray, shape (L,)
    """
    # ---- 1) Inference on TEST (normalized) ----
    preds_test, gts_test = _run_inference(model, test_loader, device)
    N_test, C, H_out, W_out = preds_test.shape

    input_len  = cfg.H_in * cfg.W_in
    output_len = cfg.H_out * cfg.W_out
    stride     = cfg.stride

    test_idx = np.asarray(test_idx)

    # ---- 2) Reconstruct full normalized series in raw index space ----
    max_g = int(test_idx.max())
    total_raw_len = max_g * stride + input_len + output_len

    pred_sum = np.zeros(total_raw_len, dtype=np.float64)
    gt_sum   = np.zeros(total_raw_len, dtype=np.float64)
    count    = np.zeros(total_raw_len, dtype=np.int32)

    for i in range(N_test):
        g = int(test_idx[i])
        out_start = g * stride + input_len
        out_end   = out_start + output_len

        pred_flat = preds_test[i, 0].reshape(-1)  # (output_len,)
        gt_flat   = gts_test[i, 0].reshape(-1)

        pred_sum[out_start:out_end] += pred_flat
        gt_sum[out_start:out_end]   += gt_flat
        count[out_start:out_end]    += 1

    mask = count > 0
    if not np.any(mask):
        raise RuntimeError("[reconstruct_series_original_scale] No forecast positions in TEST region.")

    pred_full_norm = np.zeros_like(pred_sum)
    gt_full_norm   = np.zeros_like(gt_sum)
    pred_full_norm[mask] = pred_sum[mask] / count[mask]
    gt_full_norm[mask]   = gt_sum[mask]   / count[mask]

    # compressed sequences (only where predictions exist)
    compressed_pred_norm = pred_full_norm[mask]  # (L,)
    compressed_gt_norm   = gt_full_norm[mask]    # (L,)

    # ---- 3) Inverse-transform to ORIGINAL scale ----
    compressed_pred_orig = scaler.inverse_transform(
        compressed_pred_norm.reshape(-1, 1)
    ).flatten()
    compressed_gt_orig = scaler.inverse_transform(
        compressed_gt_norm.reshape(-1, 1)
    ).flatten()

    return compressed_pred_orig, compressed_gt_orig


def visualize_test_windows_multi_models_original(
    series_dict,
    seq_len: int = 5,
    num_windows: int = 3,
):
    """
    Plot several TEST prediction windows (GT vs multiple models) in ORIGINAL scale,
    all on the SAME PLOT per window.

    series_dict: dict
        {
          "unet":    (pred_unet,    gt_unet),
          "cnn_gru": (pred_cnn_gru, gt_cnn_gru),
          "cnn_lstm":(pred_cnn_lstm,gt_cnn_lstm),
        }
        Each pred_* and gt_* is a 1D np.array of length L_model.

    seq_len: how many non-overlap horizons per window (same as in single-model viz).
    num_windows: how many different windows to visualize.

    NOTE: We align all models by truncating to the MINIMUM common length L_min.
    This ignores slight differences in absolute position in the raw series, but
    keeps them comparable as time series shapes.
    """
    os.makedirs(cfg.visualized_dir, exist_ok=True)

    # 1) Compute common length across models, and pick GT from the first entry
    model_names = list(series_dict.keys())
    if len(model_names) == 0:
        print("[MULTI_VIZ] series_dict is empty.")
        return

    # Truncate all series to the same length
    lengths = [len(series_dict[name][0]) for name in model_names]
    L_min = min(lengths)

    print(f"[MULTI_VIZ] Models: {model_names}")
    print(f"[MULTI_VIZ] Individual lengths: {lengths}, using L_min={L_min}")

    truncated = {}
    for name in model_names:
        pred_orig, gt_orig = series_dict[name]
        truncated[name] = (pred_orig[:L_min], gt_orig[:L_min])

    # Use GT from the first model as reference (they should be the same in principle)
    ref_gt = truncated[model_names[0]][1]

    # 2) Determine window + horizon sizes in samples
    horizon_len_samples = (cfg.H_out - cfg.overlap_rows) * cfg.W_out
    window_len_samples  = horizon_len_samples * seq_len

    if window_len_samples > L_min:
        print(
            f"[MULTI_VIZ] Not enough predicted length (L_min={L_min}) "
            f"for window_len={window_len_samples}."
        )
        return

    window_seconds = window_len_samples * cfg.sample_interval_sec
    window_minutes = window_seconds / 60.0

    print(
        f"[MULTI_VIZ] horizon_len={horizon_len_samples} samples, "
        f"seq_len={seq_len} -> window_len={window_len_samples} samples "
        f"= {window_seconds:.0f} sec = {window_minutes:.1f} min"
    )

    # 3) Choose start positions in [0 .. L_min-window_len_samples]
    max_start = L_min - window_len_samples
    num_windows = min(num_windows, max_start + 1)
    starts = np.linspace(0, max_start, num_windows, dtype=int)

    # y-axis limit across all models & GT
    all_vals = [ref_gt]
    for name in model_names:
        all_vals.append(truncated[name][0])
    all_vals = np.concatenate(all_vals)
    ymax = all_vals.max()

    # optional: fixed colors or let matplotlib choose
    # Model label -> style
    styles = {
        "unet":     dict(linestyle="--", linewidth=2.2, color='red'),
        "cnn_gru":  dict(linestyle=":", linewidth=2.2,  color='green'),
        "cnn_lstm": dict(linestyle="-.", linewidth=2,  color='grey'),
    }
    # styles = {
    #     "unet": dict(linestyle="-.", linewidth=2.2, color='red'),
    #     "cnn_gru": dict(linestyle="--", linewidth=2.2),
    #     "cnn_lstm": dict(linestyle=":", linewidth=2.2),
    # }

    # 4) Plot each window
    for j, start_idx in enumerate(starts):
        end_idx = start_idx + window_len_samples

        gt_seg = ref_gt[start_idx:end_idx]
        time_axis_sec = np.arange(window_len_samples) * cfg.sample_interval_sec

        plt.figure(figsize=(10, 4))

        # Ground Truth in bold
        plt.plot(
            time_axis_sec,
            gt_seg,
            label="Ground Truth",
            linewidth=2,
            color="blue",
        )

        # Predictions from each model
        for name in model_names:
            pred_seg = truncated[name][0][start_idx:end_idx]
            style = styles.get(name, {})
            plt.plot(
                time_axis_sec,
                pred_seg,
                label=f"Predictions ({name})",
                **style,
            )

        plt.xlabel("Time within window (seconds, starting at 0)")
        plt.ylabel("Traffic volume (original)")
        plt.title(
            f"TEST multi-model window #{j} "
            f"– duration={window_seconds:.0f}s ({window_minutes:.1f} min)"
        )
        plt.grid(True)
        plt.legend()
        plt.ylim(0, ymax * 1.05)
        plt.tight_layout()

        fname = f"test_multi_models_window_{j}_len{window_len_samples}_{cfg.image_tag}.png"
        out_path = os.path.join(cfg.visualized_dir, fname)
        plt.savefig(out_path, dpi=150)
        plt.show()
        plt.close()

        print(f"[MULTI_VIZ] Saved window #{j} to {out_path}")


def main():
    device = cfg.device
    if not torch.cuda.is_available():
        device = "cpu"
    print(f"Using device: {device}")

    # Which models you want to compare
    model_types = ["unet", "cnn_gru", "cnn_lstm"]

    series_dict = {}

    for mtype in model_types:
        # 1) Load model + test loader + scaler + test_idx
        model, test_loader, scaler, test_idx = load_model_and_data_for_type(
            model_type=mtype,
            device=device,
        )

        # 2) Reconstruct 1D series (original scale)
        print(f"[CoreGAN] Reconstructing series for {mtype} ...")
        pred_orig, gt_orig = reconstruct_series_original_scale(
            model=model,
            test_loader=test_loader,
            test_idx=test_idx,
            device=device,
            scaler=scaler,
        )

        series_dict[mtype] = (pred_orig, gt_orig)

    # 3) Plot multi-model overlay
    visualize_test_windows_multi_models_original(
        series_dict=series_dict,
        seq_len=cfg.context_len,  # typically 5
        num_windows=20,            # or 1 if you want just one
    )


if __name__ == "__main__":
    main()



