# utils/image_transform.py

import numpy as np
try:
    from scipy import signal as _scipy_signal
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def _seq2seq_gray2d(series_norm: np.ndarray,
                    H_in: int, W_in: int,
                    H_out: int, W_out: int,
                    overlap_len: int,
                    stride: int):
    """
    Core function: seq2seq grayscale 2D images with flexible overlap and stride.
    Returns:
        X: (N, H_in, W_in)
        Y: (N, H_out, W_out)
    """
    series_norm = np.asarray(series_norm).flatten()

    input_len = H_in * W_in
    output_len = H_out * W_out

    if overlap_len < 0 or overlap_len > min(input_len, output_len):
        raise ValueError(
            f"overlap_len must be in [0, {min(input_len, output_len)}], "
            f"got {overlap_len}"
        )

    if stride <= 0:
        raise ValueError("stride must be positive")

    X_list, Y_list = [], []

    max_start = len(series_norm) - (input_len + output_len - overlap_len)
    if max_start <= 0:
        raise ValueError(
            "Time series too short for given window sizes and overlap. "
            f"len={len(series_norm)}, input_len={input_len}, "
            f"output_len={output_len}, overlap_len={overlap_len}"
        )

    start = 0
    while start <= max_start:
        # Input window
        x = series_norm[start: start + input_len]

        # Output window with overlap at the end of the input
        out_start = start + input_len - overlap_len
        y = series_norm[out_start: out_start + output_len]

        x_img = x.reshape(H_in, W_in)
        y_img = y.reshape(H_out, W_out)

        X_list.append(x_img)
        Y_list.append(y_img)

        start += stride

    X = np.array(X_list, dtype=np.float32)  # (N, H_in, W_in)
    Y = np.array(Y_list, dtype=np.float32)  # (N, H_out, W_out)
    return X, Y


def build_image_pairs_gray2d(series_norm: np.ndarray,
                             H_in: int, W_in: int,
                             H_out: int, W_out: int,
                             overlap_len: int,
                             stride: int):
    """
    Wrapper for grayscale 2D seq2seq images.

    Returns:
        X: (N, H_in, W_in)
        Y: (N, H_out, W_out)
    """
    return _seq2seq_gray2d(
        series_norm=series_norm,
        H_in=H_in, W_in=W_in,
        H_out=H_out, W_out=W_out,
        overlap_len=overlap_len,
        stride=stride,
    )


def build_image_pairs_rgb2d(series_norm: np.ndarray,
                            H_in: int, W_in: int,
                            H_out: int, W_out: int,
                            overlap_len: int,
                            stride: int):
    """
    Example RGB construction.

    Current implementation:
      1) Build grayscale seq2seq images.
      2) Replicate grayscale channel 3 times -> fake RGB.

    Later you can change this to more meaningful RGB channels, e.g.:
       - channel 0: current window
       - channel 1: moving average / smoothed trend
       - channel 2: high-pass / residual
    """
    X_gray, Y_gray = _seq2seq_gray2d(
        series_norm=series_norm,
        H_in=H_in, W_in=W_in,
        H_out=H_out, W_out=W_out,
        overlap_len=overlap_len,
        stride=stride,
    )

    # (N, H, W) -> (N, 3, H, W)
    X_rgb = np.stack([X_gray, X_gray, X_gray], axis=1)
    Y_rgb = np.stack([Y_gray, Y_gray, Y_gray], axis=1)

    return X_rgb.astype(np.float32), Y_rgb.astype(np.float32)


# -----------------------------
# Spectrogram helper functions
# -----------------------------
def _resize_2d(img: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Simple resize helper using cropping or zero-padding
    (no fancy interpolation to keep dependencies light).

    img: (h, w)
    returns: (H, W)
    """
    h, w = img.shape

    # Crop or pad frequency axis
    if h > H:
        freq_sel = img[:H, :]
    else:
        freq_sel = np.zeros((H, w), dtype=img.dtype)
        freq_sel[:h, :] = img

    # Crop or pad time axis
    if w > W:
        out = freq_sel[:, :W]
    else:
        out = np.zeros((H, W), dtype=img.dtype)
        out[:, :w] = freq_sel

    return out


def _spectrogram_window(x_1d: np.ndarray,
                        H: int,
                        W: int,
                        nperseg: int = None,
                        noverlap: int = None) -> np.ndarray:
    """
    Build a log-magnitude spectrogram for a 1D window and resize to (H, W).

    Uses scipy.signal.spectrogram if available. If scipy is missing,
    this will raise ImportError with a clear message.
    """
    if not _HAVE_SCIPY:
        raise ImportError(
            "Spectrogram mode requires scipy. "
            "Please install it via `pip install scipy`."
        )

    x_1d = np.asarray(x_1d, dtype=np.float32).flatten()

    # Reasonable defaults: if user didn't specify, derive from H
    if nperseg is None:
        nperseg = min(len(x_1d), max(16, H))  # window length
    if noverlap is None:
        noverlap = nperseg // 2

    f, t, Sxx = _scipy_signal.spectrogram(
        x_1d,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    # Sxx: (freq_bins, time_bins)
    Sxx = np.log1p(Sxx)  # log(1 + |X|) for better dynamic range

    # Normalize spectrogram to [0,1] per window (optional)
    if Sxx.size > 0:
        S_min = Sxx.min()
        S_max = Sxx.max()
        if S_max > S_min:
            Sxx = (Sxx - S_min) / (S_max - S_min)

    # Resize to target HxW via simple crop/pad
    S_resized = _resize_2d(Sxx, H, W)
    return S_resized.astype(np.float32)


def build_image_pairs_spectrogram(series_norm: np.ndarray,
                                  H_in: int, W_in: int,
                                  H_out: int, W_out: int,
                                  overlap_len: int,
                                  stride: int):
    """
    Spectrogram-based seq2seq images.

    For each input/output 1D window (same as in gray2d),
    we compute a spectrogram and then resize it to (H_in,W_in) / (H_out,W_out).

    Returns:
        X_spec: (N, H_in, W_in)
        Y_spec: (N, H_out, W_out)
    """
    series_norm = np.asarray(series_norm).flatten()

    input_len = H_in * W_in
    output_len = H_out * W_out

    if overlap_len < 0 or overlap_len > min(input_len, output_len):
        raise ValueError(
            f"overlap_len must be in [0, {min(input_len, output_len)}], "
            f"got {overlap_len}"
        )

    if stride <= 0:
        raise ValueError("stride must be positive")

    X_list, Y_list = [], []

    max_start = len(series_norm) - (input_len + output_len - overlap_len)
    if max_start <= 0:
        raise ValueError(
            "Time series too short for given window sizes and overlap. "
            f"len={len(series_norm)}, input_len={input_len}, "
            f"output_len={output_len}, overlap_len={overlap_len}"
        )

    start = 0
    while start <= max_start:
        # 1D input and output windows
        x = series_norm[start: start + input_len]
        out_start = start + input_len - overlap_len
        y = series_norm[out_start: out_start + output_len]

        # Spectrogram for input / output
        x_spec = _spectrogram_window(x, H_in, W_in)
        y_spec = _spectrogram_window(y, H_out, W_out)

        X_list.append(x_spec)
        Y_list.append(y_spec)

        start += stride

    X_spec = np.array(X_list, dtype=np.float32)  # (N, H_in, W_in)
    Y_spec = np.array(Y_list, dtype=np.float32)  # (N, H_out, W_out)
    return X_spec, Y_spec


def build_image_pairs(series_norm: np.ndarray,
                      mode: str,
                      H_in: int, W_in: int,
                      H_out: int, W_out: int,
                      overlap_len: int,
                      stride: int):
    """
    Dispatcher: choose image construction mode.

    Returns:
        X, Y
        - gray2d: X (N, H_in, W_in), Y (N, H_out, W_out)
        - rgb2d:  X (N, 3, H_in, W_in), Y (N, 3, H_out, W_out)
        - spectrogram: TBD
    """
    mode = mode.lower()
    if mode == "gray2d":
        return build_image_pairs_gray2d(
            series_norm, H_in, W_in, H_out, W_out, overlap_len, stride
        )
    elif mode == "rgb2d":
        return build_image_pairs_rgb2d(
            series_norm, H_in, W_in, H_out, W_out, overlap_len, stride
        )
    elif mode == "spectrogram":
        return build_image_pairs_spectrogram(
            series_norm, H_in, W_in, H_out, W_out, overlap_len, stride
        )
    else:
        raise ValueError(f"Unknown image mode: {mode}")
