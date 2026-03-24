import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal


def butterworth_filter_1d(col, frame_rate, order=2, cutoff=5, ftype="low"):
    """
    1D Zero-phase Butterworth filter (dual pass).
    Handles NaNs and zero values gracefully.

    Parameters
    ----------
    col : numpy.ndarray or pandas.Series
        Input data sequence.
    frame_rate : int
        Frame rate of the recording.
    order : int, optional
        Filter order. Default is 2.
    cutoff : float, optional
        Cutoff frequency in Hz. Default is 5.
    ftype : str, optional
        Filter type (low, high, band). Default is 'low'.

    Returns
    -------
    col_filtered : pandas.Series
        Filtered sequence.
    """
    b, a = signal.butter(order // 2, cutoff / (frame_rate / 2), ftype, analog=False)
    padlen = 3 * max(len(a), len(b))

    col_filtered = col.copy()
    mask = np.isnan(col_filtered) | col_filtered.eq(0)
    falsemask_indices = np.where(~mask)[0]
    gaps = np.where(np.diff(falsemask_indices) > 1)[0] + 1
    idx_sequences = np.split(falsemask_indices, gaps)

    for seq in idx_sequences:
        if len(seq) > padlen:
            col_filtered[seq] = signal.filtfilt(b, a, col_filtered[seq])

    return col_filtered


def filter_file(input_path, save_path, order=2, cutoff=5, ftype="low", frame_rate=50):
    """
    Filter a TRC or CSV file and save filtered version.

    Parameters
    ----------
    input_path : str
        Path to the input TRC or CSV file.
    save_path : str
        Path to save the filtered file.
    order : int
        Butterworth filter order.
    cutoff : float
        Cutoff frequency in Hz.
    ftype : str
        Filter type.
    frame_rate : int or None
        Frame rate. If None, will try to infer from video in same directory.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in [".trc", ".csv"]:
        raise ValueError("Input file must be .trc or .csv")

    delimiter = "\t" if ext == ".trc" else ","

    # Read file
    with open(input_path, "r") as f:
        header = [next(f) for _ in range(5)] if ext == ".trc" else []

    df = pd.read_csv(input_path, sep=delimiter, skiprows=4 if ext == ".trc" else 0)
    frames_col, time_col = df.iloc[:, 0], df.iloc[:, 1]
    Q_coord = df.drop(df.columns[[0, 1]], axis=1)

    # Apply filtering
    Q_filt = Q_coord.apply(
        butterworth_filter_1d, axis=0, args=(frame_rate, order, cutoff, ftype)
    )
    Q_filt.insert(0, "Frame#", frames_col)
    Q_filt.insert(1, "Time", time_col)

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        [f.write(line) for line in header]
        Q_filt.to_csv(f, sep=delimiter, index=False, header=None, lineterminator="\n")


def main():
    parser = argparse.ArgumentParser(
        description="Butterworth filtering for TRC/CSV motion capture files."
    )
    parser.add_argument(
        "--input", "-i", help="Path to input TRC or CSV file", required=True
    )
    parser.add_argument(
        "--output", "-o", help="Path to save filtered output", required=True
    )
    parser.add_argument("--order", type=int, default=2, help="Filter order (default=2)")
    parser.add_argument(
        "--cutoff", type=float, default=5, help="Cutoff frequency in Hz (default=5)"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="low",
        choices=["low", "high", "band"],
        help="Filter type (default=low)",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=50,
        help="Frame rate of recording. Default is 50.",
    )
    args = parser.parse_args()

    filter_file(
        args.input,
        args.output,
        order=args.order,
        cutoff=args.cutoff,
        ftype=args.type,
        frame_rate=args.frame_rate,
    )


if __name__ == "__main__":
    sys.exit(main())
