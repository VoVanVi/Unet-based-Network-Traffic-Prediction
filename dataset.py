# dataset.py

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

from config import cfg
from utils.image_transform import build_image_pairs


# --------------------------
# Helper: dataset path name
# --------------------------

def _get_dataset_path():
    os.makedirs(cfg.processed_dir, exist_ok=True)
    fname = f"dataset_{cfg.image_mode}_{cfg.image_tag}.npz"
    return os.path.join(cfg.processed_dir, fname)


def _build_sequences(X, Y, K):
    """
    Build sequences of K consecutive windows:
      X: (N, C, H_in, W_in)
      Y: (N, H_out, W_out) or (N, C_out, H_out, W_out)

    Returns:
      X_seq:      (N_seq, K, C, H_in, W_in)
      Y_seq:      (N_seq, H_out, W_out) or (N_seq, C_out, H_out, W_out)
      orig_idx:   (N_seq,) original window index t (0..N-1) for each Y
                  (we use t as the anchor for reconstruction).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    N = X.shape[0]
    assert N == Y.shape[0], "X and Y must have same length"

    if N < K:
        raise ValueError(f"Not enough windows ({N}) to build sequences with K={K}")

    X_seq_list = []
    Y_list = []
    idx_list = []

    # t is index of the *target* window
    for t in range(K - 1, N):
        # windows [t-K+1, ..., t] are the context
        X_seq_list.append(X[t - K + 1 : t + 1])  # (K, C, H_in, W_in)
        Y_list.append(Y[t])                      # target horizon
        idx_list.append(t)                       # original window index for Y

    X_seq = np.stack(X_seq_list, axis=0)        # (N_seq, K, C, H_in, W_in)
    Y_seq = np.stack(Y_list, axis=0)            # (N_seq, ..., H_out, W_out)
    orig_idx = np.array(idx_list, dtype=np.int64)

    return X_seq, Y_seq, orig_idx



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
def debug_sample_windows(values_norm, X, Y, train_idx, num_samples=2):
    """
    Debug helper: print and verify X/Y windows for a few samples.

    values_norm: full normalized 1D series
    X, Y       : constructed image windows (numpy arrays)
    train_idx  : window indices for training split
    num_samples: how many random samples to inspect
    """
    print("\n[DEBUG] Checking sample X/Y windows...\n")
    N = X.shape[0]

    # Pick random sample windows
    rng = np.random.RandomState(123)
    sample_ids = rng.choice(train_idx, size=min(num_samples, len(train_idx)), replace=False)

    input_len  = cfg.H_in  * cfg.W_in
    output_len = cfg.H_out * cfg.W_out
    stride     = cfg.stride
    overlap_len = cfg.overlap_rows * cfg.W_in

    for wid in sample_ids:
        print("--------------------------------------------------------")
        print(f"[Window ID] {wid}")

        # Calculate raw start positions exactly like build_image_pairs()
        raw_start   = wid * stride
        raw_in_end  = raw_start + input_len
        raw_out_start = raw_in_end - overlap_len
        raw_out_end   = raw_out_start + output_len

        print(f" Raw index range X : [{raw_start} ... {raw_in_end})  (len={input_len})")
        print(f" Raw index range Y : [{raw_out_start} ... {raw_out_end})  (len={output_len})")

        # Flatten X/Y to verify against the raw normalized data
        X_flat = X[wid].flatten()
        Y_flat = Y[wid].flatten()

        raw_X = values_norm[raw_start:raw_in_end]
        raw_Y = values_norm[raw_out_start:raw_out_end]

        # Basic checks
        print(f" X match raw? {np.allclose(X_flat, raw_X)}")
        print(f" Y match raw? {np.allclose(Y_flat, raw_Y)}")

        # Print last few elements
        print(f" X last 10 raw:  {raw_X[-10:]}")
        print(f" Y last 10 raw:  {raw_Y[-10:]}")
        print(f" X_img shape = {X[wid].shape}, Y_img shape = {Y[wid].shape}")

    print("\n[DEBUG] End of sample window checks\n")



# -------------------------------------
# Generate and save dataset (once)
# -------------------------------------
def generate_and_save_dataset():
    """
    Load raw CSV, normalize, construct images, split into
    train/val/test, and save to .npz + scaler to disk.
    """
    dataset_path = _get_dataset_path()
    print(f"[dataset] Generating dataset and saving to {dataset_path}")

    if not os.path.exists(cfg.raw_csv_path):
        raise FileNotFoundError(f"CSV not found at {cfg.raw_csv_path}")

    df = pd.read_csv(cfg.raw_csv_path)
    if cfg.value_col not in df.columns:
        raise KeyError(
            f"Column '{cfg.value_col}' not found in CSV. "
            f"Available: {list(df.columns)}"
        )

    values = df[cfg.value_col].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler()
    values_norm = scaler.fit_transform(values).flatten()

    # Build image pairs using utils.image_transform
    overlap_len = cfg.overlap_rows * cfg.W_in  # e.g., 24 * 24 = 576
    X, Y = build_image_pairs(
        series_norm=values_norm,
        mode=cfg.image_mode,
        H_in=cfg.H_in,
        W_in=cfg.W_in,
        H_out=cfg.H_out,
        W_out=cfg.W_out,
        overlap_len=overlap_len,
        stride=cfg.stride,
    )
    N = X.shape[0]
    print(f"[dataset] Total single windows built: {N}")

    # ---- Ensure channel dimension: (N, C, H_in, W_in) ----
    # For gray2d, build_image_pairs usually returns (N, H, W)
    if X.ndim == 3:
        # (N, H, W) -> (N, 1, H, W)
        X = X[:, None, :, :]
    elif X.ndim == 4:
        # (N, C, H, W) already
        pass
    else:
        raise ValueError(f"Unexpected X shape after build_image_pairs: {X.shape}")

    # Y is typically (N, H_out, W_out) or (N, C_out, H_out, W_out),
    # leave it as-is; TrafficImageDataset will handle channel later.

    # ---- Build sequences of K windows for CNN-GRU ----
    K = cfg.context_len
    X_seq, Y_seq, orig_idx = _build_sequences(X, Y, K)
    N_seq = X_seq.shape[0]
    print(f"[dataset] Total sequence samples: {N_seq} (context K={K})")

    # Window indices (0 .. N-1); used for time reconstruction later
    window_ids = np.arange(N)

    # Train/val/test split (by window id)
    rng = np.random.RandomState(cfg.random_seed)
    perm = rng.permutation(N)

    test_size = int(N * cfg.test_ratio)
    val_size = int(N * cfg.val_ratio)

    test_ids = perm[:test_size]
    val_ids = perm[test_size:test_size + val_size]
    train_ids = perm[test_size + val_size:]

    X_train, Y_train = X[train_ids], Y[train_ids]
    X_val, Y_val = X[val_ids], Y[val_ids]
    X_test, Y_test = X[test_ids], Y[test_ids]

    # For reconstruction, we keep the original window index for each split
    train_idx = train_ids
    val_idx = val_ids
    test_idx = test_ids

    # Chronological split based on sequence index (0..N_seq-1)
    # seq_ids = np.arange(N_seq)
    #
    # test_size = int(N_seq * cfg.test_ratio)
    # val_size = int(N_seq * cfg.val_ratio)
    # train_size = N_seq - test_size - val_size
    #
    # train_seq_ids = seq_ids[:train_size]
    # val_seq_ids = seq_ids[train_size: train_size + val_size]
    # test_seq_ids = seq_ids[train_size + val_size:]
    #
    # # These are the original window indices for each sequence's target horizon
    # train_idx = orig_idx[train_seq_ids]
    # val_idx = orig_idx[val_seq_ids]
    # test_idx = orig_idx[test_seq_ids]
    # Split X_seq and Y_seq by sequence index
    # X_train, Y_train = X_seq[train_seq_ids], Y_seq[train_seq_ids]
    # X_val, Y_val = X_seq[val_seq_ids], Y_seq[val_seq_ids]
    # X_test, Y_test = X_seq[test_seq_ids], Y_seq[test_seq_ids]

    # After X_train, Y_train, train_idx are computed
    debug_sample_windows(values_norm, X, Y, train_idx, num_samples=2)

    # Save dataset arrays + meta
    np.savez_compressed(
        dataset_path,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        X_test=X_test,
        Y_test=Y_test,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        H_in=cfg.H_in,
        W_in=cfg.W_in,
        H_out=cfg.H_out,
        W_out=cfg.W_out,
        stride=cfg.stride,
        K=K,
        image_mode=cfg.image_mode,
    )

    # Save scaler
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    joblib.dump(scaler, cfg.scaler_path)
    print(f"[dataset] Saved scaler to {cfg.scaler_path}")


# -------------------------------------
# Load dataset from disk
# -------------------------------------
def _load_dataset_arrays():
    """
    Load dataset arrays and meta from .npz.
    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test,
        train_idx, val_idx, test_idx
    """
    dataset_path = _get_dataset_path()
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


# -------------------------------------
# Public API: prepare_dataloaders()
# -------------------------------------
def prepare_dataloaders():
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
    dataset_path = _get_dataset_path()
    if not os.path.exists(dataset_path):
        generate_and_save_dataset()
    else:
        print(f"[dataset] Using cached dataset: {dataset_path}")

    # Load arrays
    X_train, Y_train, X_val, Y_val, X_test, Y_test, train_idx, val_idx, test_idx = (
        _load_dataset_arrays()
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


# dataset.py (optional debug main)

if __name__ == "__main__":
    print("[dataset] Forcing dataset regeneration...")
    generate_and_save_dataset()
    print("[dataset] Done. File saved at:")
    print(_get_dataset_path())
