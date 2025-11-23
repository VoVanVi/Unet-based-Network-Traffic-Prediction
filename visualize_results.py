# visualize_results.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.gridspec as gridspec

from config import cfg


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


# =========================================================
# 1) IMAGE-LEVEL VISUALIZATION (a few test samples)
# =========================================================
from typing import Optional
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from config import cfg


def visualize_sample_images(
    model,
    test_loader,
    device,
    scaler,
    num_samples: int = 3,
    selection: str = "best",          # "best", "worst", or "random"
    test_idx: Optional[np.ndarray] = None,   # global window indices for test set
):
    """
    Select test samples based on prediction quality (MAE + MSE) and visualize them.

    Uses the shared `_run_inference` to:
      1) Run model over entire test_loader (normalized).
      2) Compute per-sample MAE and MSE over (C,H,W).
      3) Select 'num_samples' samples according to `selection`:
         - "best":  smallest *combined* rank of MAE and MSE
         - "worst": largest combined rank
         - "random": random subset
      4) For each selected sample:
         - show GT, Pred, |Error| (normalized) as images
         - show GT vs Pred as 1D curve in ORIGINAL scale
         - if test_idx is provided, also show global window id + horizon raw indices.
    """
    os.makedirs(cfg.visualized_dir, exist_ok=True)

    # 1) Run inference over ALL test windows (normalized space)
    preds_test, gts_test = _run_inference(model, test_loader, device)
    N, C, H, W = preds_test.shape

    # 2) Per-sample MAE and MSE
    diff = preds_test - gts_test                     # (N, C, H, W)
    abs_diff = np.abs(diff)
    sq_diff = diff ** 2

    mae_per_sample = abs_diff.mean(axis=(1, 2, 3))   # (N,)
    mse_per_sample = sq_diff.mean(axis=(1, 2, 3))    # (N,)

    # 3) Build ordering according to selection
    selection = selection.lower()
    if selection in ("best", "worst"):
        # Rank (0 = best/smallest) for each metric
        mae_rank = np.argsort(np.argsort(mae_per_sample))
        mse_rank = np.argsort(np.argsort(mse_per_sample))
        combined_rank = mae_rank + mse_rank  # “good” if both small

        if selection == "best":
            order = np.argsort(combined_rank)      # ascending: best first
        else:  # "worst"
            order = np.argsort(-combined_rank)     # descending: worst first

    elif selection == "random":
        order = np.random.permutation(N)
    else:
        raise ValueError(f"Unknown selection mode: {selection}")

    chosen = order[: min(num_samples, N)]

    # helpful constants for horizon ranges
    input_len   = cfg.H_in * cfg.W_in
    output_len  = cfg.H_out * cfg.W_out
    stride      = cfg.stride
    overlap_len = cfg.overlap_rows * cfg.W_in  # if you use overlap_rows

    # Ensure test_idx matches N if provided
    if test_idx is not None:
        test_idx = np.asarray(test_idx)
        if len(test_idx) != N:
            print(
                f"[WARN] len(test_idx)={len(test_idx)} != N_test={N}; "
                f"ignoring test_idx for sample visualization."
            )
            test_idx = None

    for rank, sample_idx in enumerate(chosen):
        pred_img_norm = preds_test[sample_idx, 0]  # (H, W)
        gt_img_norm   = gts_test[sample_idx, 0]

        sample_mae = float(mae_per_sample[sample_idx])
        sample_mse = float(mse_per_sample[sample_idx])

        # --- global window id & horizon indices (if test_idx is available) ---
        if test_idx is not None:
            global_win_id = int(test_idx[sample_idx])

            raw_in_start    = global_win_id * stride
            raw_in_end      = raw_in_start + input_len
            raw_out_start   = raw_in_end - overlap_len
            raw_out_end     = raw_out_start + output_len
            horizon_start_raw = raw_out_start
            horizon_end_raw   = raw_out_end - 1
        else:
            global_win_id = None
            horizon_start_raw = None
            horizon_end_raw   = None

        # --- Flatten & inverse-transform to original scale ---
        pred_flat_norm = pred_img_norm.reshape(-1, 1)
        gt_flat_norm   = gt_img_norm.reshape(-1, 1)

        pred_flat_orig = scaler.inverse_transform(pred_flat_norm).flatten()
        gt_flat_orig   = scaler.inverse_transform(gt_flat_norm).flatten()

        T = pred_flat_orig.shape[0]
        time_axis_sec = np.arange(T) * cfg.sample_interval_sec
        time_axis_min = time_axis_sec / 60.0

        err_img_norm = np.abs(pred_img_norm - gt_img_norm)

        # ---- Figure with 2 rows: images + time series ----
        fig = plt.figure(figsize=(12, 6))
        gs  = gridspec.GridSpec(2, 3, height_ratios=[2, 1])

        title_lines = [
            f"Test sample (selection rank {rank}) – {cfg.image_tag}",
            f"Selection mode: {selection}",
            f"MAE = {sample_mae:.6f}, MSE = {sample_mse:.6f}",
            f"Horizon length: {T} steps = {T * cfg.sample_interval_sec / 60:.1f} min",
        ]
        if global_win_id is not None and horizon_start_raw is not None:
            title_lines.append(
                f"Global window id = {global_win_id}, "
                f"horizon raw idx [{horizon_start_raw} .. {horizon_end_raw}]"
            )

        fig.suptitle("\n".join(title_lines), fontsize=10)

        # Row 0: images (normalized)
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(gt_img_norm, aspect="auto")
        ax1.set_title("Ground Truth (normalized)")
        ax1.axis("off")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(pred_img_norm, aspect="auto")
        ax2.set_title("Predictions (normalized)")
        ax2.axis("off")
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(err_img_norm, aspect="auto")
        ax3.set_title("|Error| (normalized)")
        ax3.axis("off")
        fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Row 1: original-scale time series
        ax4 = fig.add_subplot(gs[1, :])
        ax4.plot(time_axis_min, gt_flat_orig, label="Ground Truth", linewidth=2)
        ax4.plot(
            time_axis_min,
            pred_flat_orig,
            label="Predictions",
            linestyle="--",
            linewidth=1.5,
        )
        ax4.set_xlabel(
            f"Time within horizon (minutes, Δt = {cfg.sample_interval_sec}s)"
        )
        ax4.set_ylabel("Traffic volume (original)")
        ax4.grid(True)
        ax4.legend()

        fig.tight_layout(rect=[0, 0.03, 1, 0.92])

        # filename encodes selection + sample index + MAE+MSE + optional global id
        base = (
            f"sample_rank{rank}_idx{sample_idx}_"
            f"mae{sample_mae:.6f}_mse{sample_mse:.6f}_{cfg.image_tag}"
        )
        if global_win_id is not None:
            fname = f"{base}_win{global_win_id}.png"
        else:
            fname = base + ".png"

        out_path = os.path.join(cfg.visualized_dir, fname)
        fig.savefig(out_path, dpi=150)
        plt.show()
        plt.close(fig)

        print(f"Saved sample visualization to {out_path}")


# def visualize_sample_images(
#     model,
#     test_loader,
#     device,
#     scaler,
#     num_samples=3,
# ):
#     """
#     For a few test samples, visualize:
#       - GT, Pred, |Error| as images (normalized)
#       - GT vs Pred as 1D curve in ORIGINAL scale
#
#     Saves files with cfg.image_tag in the name.
#     """
#     os.makedirs(cfg.results_dir, exist_ok=True)
#     model.eval()
#
#     batch = next(iter(test_loader))
#     X, Y = batch
#     X = X.to(device)
#     Y = Y.to(device)
#
#     with torch.no_grad():
#         pred = model(X)
#
#     pred_np = pred.cpu().numpy()  # (B, C, H, W)
#     Y_np = Y.cpu().numpy()
#
#     B, C, H, W = pred_np.shape
#     n_plot = min(num_samples, B)
#
#     for k in range(n_plot):
#         pred_img_norm = pred_np[k, 0]  # (H, W)
#         gt_img_norm = Y_np[k, 0]
#
#         # Flatten and inverse transform to original scale
#         pred_flat_norm = pred_img_norm.reshape(-1, 1)
#         gt_flat_norm = gt_img_norm.reshape(-1, 1)
#
#         pred_flat_orig = scaler.inverse_transform(pred_flat_norm).flatten()
#         gt_flat_orig = scaler.inverse_transform(gt_flat_norm).flatten()
#
#         T = pred_flat_orig.shape[0]
#         time_axis_sec = np.arange(T) * cfg.sample_interval_sec
#         time_axis_min = time_axis_sec / 60.0
#
#         err_img_norm = np.abs(pred_img_norm - gt_img_norm)
#
#         # ---- Figure with 2 rows: images + time series ----
#         fig = plt.figure(figsize=(12, 6))
#         gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])
#
#         fig.suptitle(
#             f"Test sample #{k} – {cfg.image_tag}\n"
#             f"Horizon length: {T} steps = {T * cfg.sample_interval_sec / 60:.1f} min",
#             fontsize=10,
#         )
#
#         # Row 0: images (normalized)
#         ax1 = fig.add_subplot(gs[0, 0])
#         im1 = ax1.imshow(gt_img_norm, aspect="auto")
#         ax1.set_title("Ground Truth (normalized)")
#         ax1.axis("off")
#         fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
#
#         ax2 = fig.add_subplot(gs[0, 1])
#         im2 = ax2.imshow(pred_img_norm, aspect="auto")
#         ax2.set_title("Predictions (normalized)")
#         ax2.axis("off")
#         fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
#
#         ax3 = fig.add_subplot(gs[0, 2])
#         im3 = ax3.imshow(err_img_norm, aspect="auto")
#         ax3.set_title("|Error| (normalized)")
#         ax3.axis("off")
#         fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
#
#         # Row 1: original-scale time series
#         ax4 = fig.add_subplot(gs[1, :])
#         ax4.plot(time_axis_min, gt_flat_orig, label="Ground Truth", linewidth=2)
#         ax4.plot(time_axis_min, pred_flat_orig, label="Predictions", linestyle="--", linewidth=1.5,)
#         ax4.set_xlabel(
#             f"Time within horizon (minutes, Δt = {cfg.sample_interval_sec}s)"
#         )
#         ax4.set_ylabel("Traffic volume (original)")
#         ax4.grid(True)
#         ax4.legend()
#
#         fig.tight_layout(rect=[0, 0.03, 1, 0.92])
#
#         out_path = os.path.join(
#             cfg.visualized_dir,
#             f"sample_{k}_{cfg.image_tag}.png",
#         )
#         fig.savefig(out_path, dpi=150)
#         plt.show()
#         plt.close(fig)
#
#         print(f"Saved sample visualization to {out_path}")


# =========================================================
# 2) TRAFFIC OVER ALL TEST HORIZONS
# =========================================================
def visualize_all_test_traffic(model, test_loader, test_idx, device, scaler):
    """
    Reconstruct a 1D time series (normalized, then inverse-transformed)
    for GT and prediction over the TEST horizons only, and plot them.
    We take into account overlapping horizons by averaging where multiple
    predictions fall on the same raw time step.
    """
    os.makedirs(cfg.results_dir, exist_ok=True)

    # Run inference on TEST
    preds_test, gts_test = _run_inference(model, test_loader, device)
    N_test, C, H_out, W_out = preds_test.shape

    input_len = cfg.H_in * cfg.W_in
    output_len = cfg.H_out * cfg.W_out
    stride = cfg.stride

    # test_idx[i] is the global window index in [0, N_total)
    test_idx = np.asarray(test_idx)
    max_g = int(test_idx.max())
    total_raw_len = max_g * stride + input_len + output_len

    pred_sum = np.zeros(total_raw_len, dtype=np.float64)
    gt_sum = np.zeros(total_raw_len, dtype=np.float64)
    count = np.zeros(total_raw_len, dtype=np.int32)

    for i in range(N_test):
        g = int(test_idx[i])
        out_start = g * stride + input_len
        out_end = out_start + output_len

        # Flatten horizon over channels+space; here C=1 so it's (H_out*W_out,)
        pred_flat_norm = preds_test[i, 0].reshape(-1)
        gt_flat_norm = gts_test[i, 0].reshape(-1)

        pred_sum[out_start:out_end] += pred_flat_norm
        gt_sum[out_start:out_end] += gt_flat_norm
        count[out_start:out_end] += 1

    mask = count > 0
    if not np.any(mask):
        print("No forecast positions in TEST region to visualize.")
        return

    pred_norm = np.zeros_like(pred_sum)
    gt_norm = np.zeros_like(gt_sum)
    pred_norm[mask] = pred_sum[mask] / count[mask]
    gt_norm[mask] = gt_sum[mask] / count[mask]

    # Inverse transform to original scale
    raw_idx = np.arange(total_raw_len)[mask]
    pred_orig = scaler.inverse_transform(pred_norm[mask].reshape(-1, 1)).flatten()
    gt_orig = scaler.inverse_transform(gt_norm[mask].reshape(-1, 1)).flatten()

    time_axis_sec = raw_idx * cfg.sample_interval_sec
    time_axis_min = time_axis_sec / 60.0

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis_min, gt_orig, label="Ground Truth", linewidth=1)
    plt.plot( time_axis_min, pred_orig, label="Predictions", linewidth=1, linestyle="--", alpha=0.8,)
    plt.xlabel(f"Time from dataset start (minutes, Δt = {cfg.sample_interval_sec}s)")
    plt.ylabel("Traffic volume")
    plt.title("Traffic over ALL TEST horizons (Ground Truth vs Predictions)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(cfg.visualized_dir, f"traffic_test_only_orig_{cfg.image_tag}.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    plt.close()

    print(f"Saved test traffic visualization to {out_path}")


def visualize_all_test_traffic_normalized(model, test_loader, test_idx, device):
    """
    Reconstruct a 1D time series for GT and prediction over the TEST horizons
    only, and plot them in NORMALIZED scale.
    Overlapping horizons are averaged.
    """
    os.makedirs(cfg.results_dir, exist_ok=True)

    # Run inference on TEST (normalized)
    preds_test, gts_test = _run_inference(model, test_loader, device)
    N_test, C, H_out, W_out = preds_test.shape

    input_len = cfg.H_in * cfg.W_in
    output_len = cfg.H_out * cfg.W_out
    stride = cfg.stride

    test_idx = np.asarray(test_idx)
    max_g = int(test_idx.max())
    total_raw_len = max_g * stride + input_len + output_len

    pred_sum = np.zeros(total_raw_len, dtype=np.float64)
    gt_sum = np.zeros(total_raw_len, dtype=np.float64)
    count = np.zeros(total_raw_len, dtype=np.int32)

    for i in range(N_test):
        g = int(test_idx[i])
        out_start = g * stride + input_len
        out_end = out_start + output_len

        pred_flat_norm = preds_test[i, 0].reshape(-1)
        gt_flat_norm = gts_test[i, 0].reshape(-1)

        pred_sum[out_start:out_end] += pred_flat_norm
        gt_sum[out_start:out_end] += gt_flat_norm
        count[out_start:out_end] += 1

    mask = count > 0
    if not np.any(mask):
        print("No forecast positions in TEST region to visualize.")
        return

    pred_norm = np.zeros_like(pred_sum)
    gt_norm = np.zeros_like(gt_sum)
    pred_norm[mask] = pred_sum[mask] / count[mask]
    gt_norm[mask] = gt_sum[mask] / count[mask]

    raw_idx = np.arange(total_raw_len)[mask]
    time_axis_sec = raw_idx * cfg.sample_interval_sec
    time_axis_min = time_axis_sec / 60.0

    plt.figure(figsize=(12, 4))
    plt.plot(time_axis_min, gt_norm[mask], label="Ground Truth (normalized)", linewidth=1)
    plt.plot(
        time_axis_min,
        pred_norm[mask],
        label="Predictions (normalized)",
        linewidth=1,
        linestyle="--",
        alpha=0.8,
    )
    plt.xlabel(f"Time from dataset start (minutes, Δt = {cfg.sample_interval_sec}s)")
    plt.ylabel("Normalized traffic")
    plt.title("Traffic over ALL TEST horizons (normalized Ground Truth vs Predictions)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(cfg.visualized_dir,f"traffic_test_only_norm_{cfg.image_tag}.png")
    plt.savefig(out_path, dpi=150)

    plt.show()
    plt.close()

    print(f"Saved test traffic visualization (normalized) to {out_path}")



# =========================================================
# 3) TRAFFIC OVER WHOLE DATASET (TRAIN + VAL + TEST)
# =========================================================
def visualize_full_dataset_traffic(
    model,
    train_loader,
    val_loader,
    test_loader,
    train_idx,
    val_idx,
    test_idx,
    device,
    scaler,
):
    """
    Reconstruct a 1D time series of GT and prediction over ALL forecast horizons
    (train + val + test) and plot them together.
    Regions corresponding to train/val/test forecast windows will also be marked.
    """
    os.makedirs(cfg.results_dir, exist_ok=True)

    # Run inference for each split
    preds_train, gts_train = _run_inference(model, train_loader, device)
    preds_test, gts_test = _run_inference(model, test_loader, device)

    N_train, C, H_out, W_out = preds_train.shape
    N_test = preds_test.shape[0]

    if len(val_idx) > 0:
        preds_val, gts_val = _run_inference(model, val_loader, device)
        N_val = preds_val.shape[0]
    else:
        preds_val = gts_val = None
        N_val = 0

    input_len = cfg.H_in * cfg.W_in
    output_len = cfg.H_out * cfg.W_out
    stride = cfg.stride

    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    test_idx = np.asarray(test_idx)

    all_idx = np.concatenate(
        [train_idx, val_idx, test_idx] if N_val > 0 else [train_idx, test_idx]
    )
    max_g = int(all_idx.max())
    total_raw_len = max_g * stride + input_len + output_len

    pred_sum = np.zeros(total_raw_len, dtype=np.float64)
    gt_sum = np.zeros(total_raw_len, dtype=np.float64)
    count = np.zeros(total_raw_len, dtype=np.int32)

    def accumulate(preds_split, gts_split, idx_array):
        nonlocal pred_sum, gt_sum, count
        if preds_split is None:
            return
        N_split = preds_split.shape[0]
        for i in range(N_split):
            g = int(idx_array[i])
            out_start = g * stride + input_len
            out_end = out_start + output_len

            pred_flat_norm = preds_split[i, 0].reshape(-1)
            gt_flat_norm = gts_split[i, 0].reshape(-1)

            pred_sum[out_start:out_end] += pred_flat_norm
            gt_sum[out_start:out_end] += gt_flat_norm
            count[out_start:out_end] += 1

    # accumulate all splits
    accumulate(preds_train, gts_train, train_idx)
    accumulate(preds_val, gts_val, val_idx)
    accumulate(preds_test, gts_test, test_idx)

    mask = count > 0
    if not np.any(mask):
        print("No forecast positions to visualize over full dataset.")
        return

    pred_norm = np.zeros_like(pred_sum)
    gt_norm = np.zeros_like(gt_sum)
    pred_norm[mask] = pred_sum[mask] / count[mask]
    gt_norm[mask] = gt_sum[mask] / count[mask]

    # Inverse transform
    raw_idx = np.arange(total_raw_len)[mask]
    pred_orig = scaler.inverse_transform(pred_norm[mask].reshape(-1, 1)).flatten()
    gt_orig = scaler.inverse_transform(gt_norm[mask].reshape(-1, 1)).flatten()

    time_axis_sec = raw_idx * cfg.sample_interval_sec
    time_axis_min = time_axis_sec / 60.0

    # Get approximate forecast region coverage per split
    def region_bounds(idx_array):
        if len(idx_array) == 0:
            return None
        idx_array = np.asarray(idx_array)
        start_raw = int(idx_array.min() * stride + input_len)
        end_raw = int(idx_array.max() * stride + input_len + output_len - 1)
        return start_raw, end_raw

    train_region = region_bounds(train_idx)
    val_region = region_bounds(val_idx)
    test_region = region_bounds(test_idx)

    plt.figure(figsize=(14, 5))

    # background shading in minutes
    if train_region is not None:
        s, e = train_region
        plt.axvspan(
            s * cfg.sample_interval_sec / 60.0,
            e * cfg.sample_interval_sec / 60.0,
            alpha=0.08,
            color="tab:blue",
            label="Train forecast region",
        )
    if val_region is not None:
        s, e = val_region
        plt.axvspan(
            s * cfg.sample_interval_sec / 60.0,
            e * cfg.sample_interval_sec / 60.0,
            alpha=0.08,
            color="tab:orange",
            label="Val forecast region",
        )
    if test_region is not None:
        s, e = test_region
        plt.axvspan(
            s * cfg.sample_interval_sec / 60.0,
            e * cfg.sample_interval_sec / 60.0,
            alpha=0.08,
            color="tab:green",
            label="Test forecast region",
        )

    # main curves
    plt.plot(time_axis_min, gt_orig, label="GT forecasted region", color="black", linewidth=1)
    plt.plot(
        time_axis_min,
        pred_orig,
        label="Pred forecasted region",
        color="red",
        linewidth=1,
        linestyle="--",
        alpha=0.8,
    )

    plt.xlabel(f"Time from dataset start (minutes, Δt = {cfg.sample_interval_sec}s)")
    plt.ylabel("Traffic volume")
    plt.title("Traffic over WHOLE dataset (train + val + test horizons)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(cfg.visualized_dir,f"traffic_full_dataset_orig_{cfg.image_tag}.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    plt.close()

    print(f"Saved full dataset traffic visualization to {out_path}")



def visualize_full_dataset_traffic_normalized(
    model,
    train_loader,
    val_loader,
    test_loader,
    train_idx,
    val_idx,
    test_idx,
    device,
):
    """
    Reconstruct a 1D time series of GT and prediction over ALL forecast horizons
    (train + val + test) and plot them in NORMALIZED scale.
    Regions for train/val/test forecast windows are shaded.
    """
    os.makedirs(cfg.results_dir, exist_ok=True)

    preds_train, gts_train = _run_inference(model, train_loader, device)
    preds_test, gts_test = _run_inference(model, test_loader, device)

    N_train, C, H_out, W_out = preds_train.shape
    N_test = preds_test.shape[0]

    if len(val_idx) > 0:
        preds_val, gts_val = _run_inference(model, val_loader, device)
        N_val = preds_val.shape[0]
    else:
        preds_val = gts_val = None
        N_val = 0

    input_len = cfg.H_in * cfg.W_in
    output_len = cfg.H_out * cfg.W_out
    stride = cfg.stride

    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    test_idx = np.asarray(test_idx)

    all_idx = np.concatenate(
        [train_idx, val_idx, test_idx] if N_val > 0 else [train_idx, test_idx]
    )
    max_g = int(all_idx.max())
    total_raw_len = max_g * stride + input_len + output_len

    pred_sum = np.zeros(total_raw_len, dtype=np.float64)
    gt_sum = np.zeros(total_raw_len, dtype=np.float64)
    count = np.zeros(total_raw_len, dtype=np.int32)

    def accumulate(preds_split, gts_split, idx_array):
        nonlocal pred_sum, gt_sum, count
        if preds_split is None:
            return
        N_split = preds_split.shape[0]
        idx_array = np.asarray(idx_array)
        for i in range(N_split):
            g = int(idx_array[i])
            out_start = g * stride + input_len
            out_end = out_start + output_len

            pred_flat_norm = preds_split[i, 0].reshape(-1)
            gt_flat_norm = gts_split[i, 0].reshape(-1)

            pred_sum[out_start:out_end] += pred_flat_norm
            gt_sum[out_start:out_end] += gt_flat_norm
            count[out_start:out_end] += 1

    accumulate(preds_train, gts_train, train_idx)
    accumulate(preds_val, gts_val, val_idx)
    accumulate(preds_test, gts_test, test_idx)

    mask = count > 0
    if not np.any(mask):
        print("No forecast positions to visualize over full dataset.")
        return

    pred_norm = np.zeros_like(pred_sum)
    gt_norm = np.zeros_like(gt_sum)
    pred_norm[mask] = pred_sum[mask] / count[mask]
    gt_norm[mask] = gt_sum[mask] / count[mask]

    raw_idx = np.arange(total_raw_len)[mask]
    time_axis_sec = raw_idx * cfg.sample_interval_sec
    time_axis_min = time_axis_sec / 60.0

    def region_bounds(idx_array):
        if len(idx_array) == 0:
            return None
        idx_array = np.asarray(idx_array)
        start_raw = int(idx_array.min() * stride + input_len)
        end_raw = int(idx_array.max() * stride + input_len + output_len - 1)
        return start_raw, end_raw

    train_region = region_bounds(train_idx)
    val_region = region_bounds(val_idx)
    test_region = region_bounds(test_idx)

    plt.figure(figsize=(14, 5))

    # shade forecast regions
    if train_region is not None:
        s, e = train_region
        plt.axvspan(
            s * cfg.sample_interval_sec / 60.0,
            e * cfg.sample_interval_sec / 60.0,
            alpha=0.08,
            color="tab:blue",
            label="Train forecast region",
        )
    if val_region is not None:
        s, e = val_region
        plt.axvspan(
            s * cfg.sample_interval_sec / 60.0,
            e * cfg.sample_interval_sec / 60.0,
            alpha=0.08,
            color="tab:orange",
            label="Val forecast region",
        )
    if test_region is not None:
        s, e = test_region
        plt.axvspan(
            s * cfg.sample_interval_sec / 60.0,
            e * cfg.sample_interval_sec / 60.0,
            alpha=0.08,
            color="tab:green",
            label="Test forecast region",
        )

    plt.plot(
        time_axis_min,
        gt_norm[mask],
        label="GT forecast region (norm)",
        color="black",
        linewidth=1,
    )
    plt.plot(
        time_axis_min,
        pred_norm[mask],
        label="Pred forecast region (norm)",
        color="red",
        linewidth=1,
        linestyle="--",
        alpha=0.8,
    )

    plt.xlabel(f"Time from dataset start (minutes, Δt = {cfg.sample_interval_sec}s)")
    plt.ylabel("Normalized traffic")
    plt.title("Traffic over WHOLE dataset (train + val + test, normalized)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(cfg.visualized_dir, f"traffic_full_dataset_norm_{cfg.image_tag}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved full dataset traffic visualization (normalized) to {out_path}")
