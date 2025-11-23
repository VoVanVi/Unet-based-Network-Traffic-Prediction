# data_processing_seq2seq.py

import numpy as np


def create_seq2seq_image_pairs(series_norm: np.ndarray,
                               H_in: int, W_in: int,
                               H_out: int, W_out: int,
                               overlap_len: int = 0,
                               stride: int = 1):
    """
    Create seq2seq input/output image pairs from a 1D time series with flexible overlap.

    series_norm : 1D numpy array of normalized time series.
    H_in, W_in  : height/width of input image.
    H_out, W_out: height/width of output image.
    overlap_len : number of time steps where input and output overlap.
                  0 <= overlap_len <= min(H_in*W_in, H_out*W_out)
    stride      : how many time steps to move the window each time.
                  For shift 10 s per image, use stride=1.
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

    # last usable start index
    max_start = len(series_norm) - (input_len + output_len - overlap_len)
    if max_start <= 0:
        raise ValueError(
            "Time series too short for given window sizes and overlap. "
            f"len={len(series_norm)}, input_len={input_len}, output_len={output_len}, "
            f"overlap_len={overlap_len}"
        )

    start = 0
    while start <= max_start:
        # Input window
        x = series_norm[start: start + input_len]

        # Output window with overlap at the end of the input
        out_start = start + input_len - overlap_len
        y = series_norm[out_start: out_start + output_len]

        # reshape to images
        x_img = x.reshape(H_in, W_in)
        y_img = y.reshape(H_out, W_out)

        X_list.append(x_img)
        Y_list.append(y_img)

        start += stride

    X = np.array(X_list, dtype=np.float32)  # (N, H_in, W_in)
    Y = np.array(Y_list, dtype=np.float32)  # (N, H_out, W_out)

    return X, Y
