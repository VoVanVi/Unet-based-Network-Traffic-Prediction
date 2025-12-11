# visualize_results.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib.gridspec as gridspec
import random
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
    selection: str = "best",                # "best", "worst", or "random"
    test_idx: Optional[np.ndarray] = None,  # global window indices for test set
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
         - show GT, Pred, |Error| (normalized) as images (same color scale + 1 colorbar)
         - show GT vs Pred as 1D curve in ORIGINAL scale
         - if test_idx is provided, also show global window id + horizon raw indices.
    """
    os.makedirs(cfg.visualized_dir, exist_ok=True)

    # 1) Run inference over ALL test windows (normalized space)
    preds_test, gts_test = _run_inference(model, test_loader, device)
    N, C, H, W = preds_test.shape

    # 2) Per-sample MAE and MSE (normalized)
    diff = preds_test - gts_test                     # (N, C, H, W)
    abs_diff = np.abs(diff)
    sq_diff = diff ** 2

    mae_per_sample = abs_diff.mean(axis=(1, 2, 3))   # (N,)
    mse_per_sample = sq_diff.mean(axis=(1, 2, 3))    # (N,)

    # 3) Build ordering according to selection
    selection = selection.lower()
    if selection in ("best", "worst"):
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

        # 2 rows, 4 columns:
        # row 0: [GT, Pred, Error, Colorbar]
        # row 1: [time series spanning all 4 cols]
        gs = gridspec.GridSpec(
            2, 4,
            width_ratios=[1, 1, 1, 0.05],
            height_ratios=[2, 1],
            wspace=0.3,
            hspace=0.4,
            figure=fig,   # <-- attach GridSpec to figure (no layout warning)
        )

        # --- Title text ---
        title_lines = [
            f"Test sample (selection rank {rank}) – {cfg.image_tag}",
            f"Horizon length: {T} steps = {T * cfg.sample_interval_sec / 60:.1f} min",
        ]

        # title_lines = [
        #     f"Test sample (selection rank {rank}) – {cfg.image_tag}",
        #     f"Selection mode: {selection}",
        #     f"MAE = {sample_mae:.6f}, MSE = {sample_mse:.6f}",
        #     f"Horizon length: {T} steps = {T * cfg.sample_interval_sec / 60:.1f} min",
        # ]
        # if global_win_id is not None and horizon_start_raw is not None:
        #     title_lines.append(
        #         f"Global window id = {global_win_id}, "
        #         f"horizon raw idx [{horizon_start_raw} .. {horizon_end_raw}]"
        #     )

        fig.suptitle("\n".join(title_lines), fontsize=10)

        # --- shared color scale across 3 images ---
        vmin = min(gt_img_norm.min(), pred_img_norm.min(), err_img_norm.min())
        vmax = max(gt_img_norm.max(), pred_img_norm.max(), err_img_norm.max())

        # ---- Row 0: images + shared colorbar ----
        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.imshow(gt_img_norm, aspect="auto", vmin=vmin, vmax=vmax)
        ax1.set_title("Ground Truth (norm)")
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(pred_img_norm, aspect="auto", vmin=vmin, vmax=vmax)
        ax2.set_title("Prediction (norm)")
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(err_img_norm, aspect="auto", vmin=vmin, vmax=vmax)
        ax3.set_title("|Error| (norm)")
        ax3.axis("off")

        # dedicated axis for shared colorbar
        cax = fig.add_subplot(gs[0, 3])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Normalized value", fontsize=9)

        # ---- Row 1: original-scale time series (span all columns) ----
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

        # leave some room for suptitle
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
    plt.plot(time_axis_min, pred_orig, label="Predictions", linewidth=1, linestyle="--", alpha=0.8,)
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


# ==========================================================
# Visualize several TEST prediction windows (GT vs Pred)
# in ORIGINAL scale with x-axis in SECONDS, starting from 0.
# ==========================================================
def visualize_test_windows_sec_original(
    model,
    test_loader,
    test_idx,
    device,
    scaler,
    seq_len: int = 5,
    num_windows: int = 3,
):
    """
    Visualize several TEST prediction windows (GT vs Pred) in ORIGINAL scale.
    - x-axis: seconds starting from 0
    - window length: (non-overlap horizon) * seq_len
      where horizon_len = (H_out - overlap_rows) * W_out (in samples)
    - y-axis: [0, max(GT, Pred)]
    """
    os.makedirs(cfg.visualized_dir, exist_ok=True)

    # 1) Inference on TEST (normalized)
    preds_test, gts_test = _run_inference(model, test_loader, device)
    N_test, C, H_out, W_out = preds_test.shape

    input_len  = cfg.H_in * cfg.W_in
    output_len = cfg.H_out * cfg.W_out
    stride     = cfg.stride

    test_idx = np.asarray(test_idx)

    # 2) Reconstruct full normalized series in raw index space
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
        print("[TEST_WINDOWS_ORIG] No forecast positions to visualize in TEST region.")
        return

    # compress to contiguous predicted region (index 0..L-1)
    pred_full_norm = np.zeros_like(pred_sum)
    gt_full_norm   = np.zeros_like(gt_sum)
    pred_full_norm[mask] = pred_sum[mask] / count[mask]
    gt_full_norm[mask]   = gt_sum[mask]   / count[mask]

    # compressed sequences (only where predictions exist)
    compressed_pred_norm = pred_full_norm[mask]  # shape (L,)
    compressed_gt_norm   = gt_full_norm[mask]    # shape (L,)

    # 3) Inverse-transform to ORIGINAL scale on compressed series
    compressed_pred_orig = scaler.inverse_transform(
        compressed_pred_norm.reshape(-1, 1)
    ).flatten()
    compressed_gt_orig = scaler.inverse_transform(
        compressed_gt_norm.reshape(-1, 1)
    ).flatten()

    L = compressed_pred_orig.shape[0]

    # 4) Non-overlap prediction horizon and window size
    horizon_len_samples = (cfg.H_out - cfg.overlap_rows) * cfg.W_out
    window_len_samples  = horizon_len_samples * seq_len

    if window_len_samples > L:
        print(
            f"[TEST_WINDOWS_ORIG] Not enough predicted length (L={L}) for window_len={window_len_samples}."
        )
        return

    window_seconds = window_len_samples * cfg.sample_interval_sec
    window_minutes = window_seconds / 60.0

    print(
        f"[TEST_WINDOWS_ORIG] horizon_len={horizon_len_samples} samples, "
        f"seq_len={seq_len} -> window_len={window_len_samples} samples "
        f"= {window_seconds:.0f} sec = {window_minutes:.1f} min"
    )

    # 5) Choose start positions in compressed index space [0..L-window_len]
    max_start = L - window_len_samples
    num_windows = min(num_windows, max_start + 1)
    starts = np.linspace(0, max_start, num_windows, dtype=int)

    # fixed y-axis [0, max]
    ymax = max(compressed_gt_orig.max(), compressed_pred_orig.max())

    # 6) Plot each window
    for j, start_idx in enumerate(starts):
        end_idx = start_idx + window_len_samples

        gt_seg   = compressed_gt_orig[start_idx:end_idx]
        pred_seg = compressed_pred_orig[start_idx:end_idx]

        # x-axis in seconds from 0
        time_axis_sec = np.arange(window_len_samples) * cfg.sample_interval_sec

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis_sec, gt_seg, label="Ground Truth", linewidth=2)
        plt.plot(
            time_axis_sec,
            pred_seg,
            label="Predictions",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
        )

        plt.xlabel("Time within window (seconds, starting at 0)")
        plt.ylabel("Traffic volume (original)")
        plt.title(
            f"TEST prediction window #{j} "
            f"– duration={window_seconds:.0f}s ({window_minutes:.1f} min)"
        )
        plt.grid(True)
        plt.legend()
        plt.ylim(0, ymax * 1.05)
        plt.tight_layout()

        fname = (
            f"test_window_orig_{j}_len{window_len_samples}_{cfg.image_tag}.png"
        )
        out_path = os.path.join(cfg.visualized_dir, fname)
        plt.savefig(out_path, dpi=150)
        plt.show()
        plt.close()

        print(f"[TEST_WINDOWS_ORIG] Saved window #{j} to {out_path}")



def visualize_test_windows_sec_normalized(
    model,
    test_loader,
    test_idx,
    device,
    seq_len: int = 5,
    num_windows: int = 3,
):
    """
    Visualize several TEST prediction windows (GT vs Pred) in NORMALIZED scale.
    - x-axis: seconds starting at 0
    - window length: (non-overlap horizon) * seq_len
    - y-axis: [0, 1]
    """
    os.makedirs(cfg.visualized_dir, exist_ok=True)

    # 1) Inference on TEST (normalized)
    preds_test, gts_test = _run_inference(model, test_loader, device)
    N_test, C, H_out, W_out = preds_test.shape

    input_len  = cfg.H_in * cfg.W_in
    output_len = cfg.H_out * cfg.W_out
    stride     = cfg.stride

    test_idx = np.asarray(test_idx)

    # 2) Reconstruct full normalized series (raw index space)
    max_g = int(test_idx.max())
    total_raw_len = max_g * stride + input_len + output_len

    pred_sum = np.zeros(total_raw_len, dtype=np.float64)
    gt_sum   = np.zeros(total_raw_len, dtype=np.float64)
    count    = np.zeros(total_raw_len, dtype=np.int32)

    for i in range(N_test):
        g = int(test_idx[i])
        out_start = g * stride + input_len
        out_end   = out_start + output_len

        pred_flat = preds_test[i, 0].reshape(-1)
        gt_flat   = gts_test[i, 0].reshape(-1)

        pred_sum[out_start:out_end] += pred_flat
        gt_sum[out_start:out_end]   += gt_flat
        count[out_start:out_end]    += 1

    mask = count > 0
    if not np.any(mask):
        print("[TEST_WINDOWS_NORM] No forecast positions to visualize in TEST region.")
        return

    pred_full_norm = np.zeros_like(pred_sum)
    gt_full_norm   = np.zeros_like(gt_sum)
    pred_full_norm[mask] = pred_sum[mask] / count[mask]
    gt_full_norm[mask]   = gt_sum[mask]   / count[mask]

    # compressed sequences
    compressed_pred_norm = pred_full_norm[mask]
    compressed_gt_norm   = gt_full_norm[mask]
    L = compressed_pred_norm.shape[0]

    # 3) Non-overlap horizon + window length
    horizon_len_samples = (cfg.H_out - cfg.overlap_rows) * cfg.W_out
    window_len_samples  = horizon_len_samples * seq_len

    if window_len_samples > L:
        print(
            f"[TEST_WINDOWS_NORM] Not enough predicted length (L={L}) for window_len={window_len_samples}."
        )
        return

    window_seconds = window_len_samples * cfg.sample_interval_sec
    window_minutes = window_seconds / 60.0

    print(
        f"[TEST_WINDOWS_NORM] horizon_len={horizon_len_samples} samples, "
        f"seq_len={seq_len} -> window_len={window_len_samples} samples "
        f"= {window_seconds:.0f} sec = {window_minutes:.1f} min"
    )

    max_start = L - window_len_samples
    num_windows = min(num_windows, max_start + 1)
    starts = np.linspace(0, max_start, num_windows, dtype=int)

    # 4) Plot windows (normalized, y in [0, 1])
    for j, start_idx in enumerate(starts):
        end_idx = start_idx + window_len_samples

        gt_seg   = compressed_gt_norm[start_idx:end_idx]
        pred_seg = compressed_pred_norm[start_idx:end_idx]

        time_axis_sec = np.arange(window_len_samples) * cfg.sample_interval_sec

        plt.figure(figsize=(10, 4))
        plt.plot(time_axis_sec, gt_seg, label="Ground Truth (norm)", linewidth=2)
        plt.plot(
            time_axis_sec,
            pred_seg,
            label="Predictions (norm)",
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
        )

        plt.xlabel("Time within window (seconds, starting at 0)")
        plt.ylabel("Normalized traffic")
        plt.title(
            f"TEST prediction window #{j} (normalized) "
            f"– duration={window_seconds:.0f}s ({window_minutes:.1f} min)"
        )
        plt.grid(True)
        plt.legend()
        plt.ylim(0, 1)   # fixed [0,1]
        plt.tight_layout()

        fname = (
            f"test_window_norm_{j}_len{window_len_samples}_{cfg.image_tag}.png"
        )
        out_path = os.path.join(cfg.visualized_dir, fname)
        plt.savefig(out_path, dpi=150)
        plt.show()
        plt.close()

        print(f"[TEST_WINDOWS_NORM] Saved window #{j} to {out_path}")





# def visualize_test_windows_sec(
#     model,
#     test_loader,
#     test_idx,
#     device,
#     scaler,
#     seq_len: int = 5,
#     num_windows: int = 3,
# ):
#     """
#     Visualize several TEST prediction windows (GT vs Pred) in ORIGINAL scale.
#
#     - x-axis is *relative* time in seconds, starting from 0 for each window.
#     - The true prediction horizon is the NON-OVERLAPPED part:
#
#           horizon_len_samples = (H_out - overlap_rows) * W_out
#
#       So in seconds:
#
#           horizon_seconds = horizon_len_samples * cfg.sample_interval_sec
#
#     - The visualization window length is:
#
#           window_len_samples = horizon_len_samples * seq_len
#
#       (i.e., 'seq_len' horizons concatenated),
#       so in seconds:
#
#           window_seconds = window_len_samples * cfg.sample_interval_sec
#
#       Example:
#         - if prediction horizon = 12 min and seq_len = 5 → 60 min window
#         - if prediction horizon = 24 min and seq_len = 5 → 120 min window
#     """
#     os.makedirs(cfg.visualized_dir, exist_ok=True)
#
#     # 1) Run inference on TEST (normalized)
#     preds_test, gts_test = _run_inference(model, test_loader, device)
#     N_test, C, H_out, W_out = preds_test.shape
#
#     input_len  = cfg.H_in * cfg.W_in
#     output_len = cfg.H_out * cfg.W_out      # total horizon image (including overlap)
#     stride     = cfg.stride
#
#     test_idx = np.asarray(test_idx)
#     if len(test_idx) != N_test:
#         print(
#             f"[TEST_WINDOWS] [WARN] len(test_idx)={len(test_idx)} != N_test={N_test}; "
#             f"test window reconstruction may be inconsistent."
#         )
#
#     # 2) Reconstruct full 1D series (normalized) over TEST forecast region
#     max_g = int(test_idx.max())
#     total_raw_len = max_g * stride + input_len + output_len
#
#     pred_sum = np.zeros(total_raw_len, dtype=np.float64)
#     gt_sum   = np.zeros(total_raw_len, dtype=np.float64)
#     count    = np.zeros(total_raw_len, dtype=np.int32)
#
#     for i in range(N_test):
#         g = int(test_idx[i])
#         out_start = g * stride + input_len
#         out_end   = out_start + output_len
#
#         pred_flat_norm = preds_test[i, 0].reshape(-1)  # (output_len,)
#         gt_flat_norm   = gts_test[i, 0].reshape(-1)
#
#         pred_sum[out_start:out_end] += pred_flat_norm
#         gt_sum[out_start:out_end]   += gt_flat_norm
#         count[out_start:out_end]    += 1
#
#     mask = count > 0
#     if not np.any(mask):
#         print("[TEST_WINDOWS] No forecast positions to visualize in TEST region.")
#         return
#
#     pred_full_norm = np.zeros_like(pred_sum)
#     gt_full_norm   = np.zeros_like(gt_sum)
#     pred_full_norm[mask] = pred_sum[mask] / count[mask]
#     gt_full_norm[mask]   = gt_sum[mask]   / count[mask]
#
#     # 3) Inverse transform to ORIGINAL scale (only values; time stays implicit)
#     gt_full_orig   = scaler.inverse_transform(gt_full_norm.reshape(-1, 1)).flatten()
#     pred_full_orig = scaler.inverse_transform(pred_full_norm.reshape(-1, 1)).flatten()
#
#     # 4) Define prediction horizon (non-overlapped) and window length
#     horizon_len_samples = (cfg.H_out - cfg.overlap_rows) * cfg.W_out
#     horizon_seconds     = horizon_len_samples * cfg.sample_interval_sec
#
#     window_len_samples = horizon_len_samples * seq_len
#     window_seconds     = window_len_samples * cfg.sample_interval_sec
#     window_minutes     = window_seconds / 60.0
#
#     print(
#         f"[TEST_WINDOWS] horizon_len={horizon_len_samples} samples "
#         f"({horizon_seconds:.0f} sec), seq_len={seq_len} -> "
#         f"window_len={window_len_samples} samples "
#         f"= {window_seconds:.0f} sec = {window_minutes:.1f} min"
#     )
#
#     # Valid region where we actually have predictions
#     valid_indices = np.where(mask)[0]
#     first_valid   = int(valid_indices.min())
#     last_valid    = int(valid_indices.max())
#
#     max_start = last_valid - window_len_samples + 1
#     if max_start <= first_valid:
#         print(
#             "[TEST_WINDOWS] Not enough predicted span for the requested window size; "
#             "try reducing seq_len or using a shorter horizon."
#         )
#         return
#
#     # 5) Choose start positions for windows (evenly spaced inside valid forecast region)
#     num_windows = min(num_windows, max_start - first_valid + 1)
#     if num_windows <= 0:
#         print("[TEST_WINDOWS] No valid windows to visualize.")
#         return
#
#     starts = np.linspace(first_valid, max_start, num_windows, dtype=int)
#
#     # 6) Plot each selected window (GT vs Pred) with x-axis starting from 0
#     for j, start_idx in enumerate(starts):
#         end_idx = start_idx + window_len_samples
#
#         gt_seg   = gt_full_orig[start_idx:end_idx]
#         pred_seg = pred_full_orig[start_idx:end_idx]
#
#         # x-axis: relative time within the window, starting from 0
#         time_axis_sec = np.arange(window_len_samples) * cfg.sample_interval_sec
#
#         plt.figure(figsize=(10, 4))
#         plt.plot(time_axis_sec, gt_seg, label="Ground Truth", linewidth=2)
#         plt.plot(
#             time_axis_sec,
#             pred_seg,
#             label="Predictions",
#             linestyle="--",
#             linewidth=1.5,
#             alpha=0.8,
#         )
#
#         plt.xlabel("Time within window (seconds)")
#         plt.ylabel("Traffic volume (original scale)")
#         plt.title(
#             f"TEST prediction window #{j} "
#             f"– duration={window_seconds:.0f}s ({window_minutes:.1f} min)\n"
#             f"horizon_len={horizon_len_samples} samples, seq_len={seq_len}"
#         )
#         plt.grid(True)
#         plt.legend()
#         plt.tight_layout()
#
#         fname = (
#             f"test_window{j}_start{start_idx}_len{window_len_samples}_"
#             f"horizon{horizon_len_samples}_seq{seq_len}_{cfg.image_tag}.png"
#         )
#         out_path = os.path.join(cfg.visualized_dir, fname)
#         plt.savefig(out_path, dpi=150)
#         plt.show()
#         plt.close()
#
#         print(f"[TEST_WINDOWS] Saved window #{j} visualization to {out_path}")



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
