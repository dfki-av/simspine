from pathlib import Path

import pandas as pd


def sto_to_trc(
    sto_path: Path,
    trc_path: Path,
    data_rate=50,
    camera_rate=50,
    units="m",
):
    # --- Read STO ---
    with open(sto_path, "r") as f:
        lines = f.readlines()

    # Find header end
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("endheader"):
            header_end = i
            break

    # Read table data
    df = pd.read_csv(sto_path, sep=r"\s+", skiprows=header_end + 1)
    n_frames, n_cols = df.shape
    n_markers = (n_cols - 1) // 3

    # --- Prepare TRC header ---
    header1 = f"PathFileType\t4\t(X/Y/Z)\t{trc_path.name}"
    header2 = "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
    header3 = f"{data_rate}\t{camera_rate}\t{n_frames}\t{n_markers}\t{units}\t{data_rate}\t0\t{n_frames}"

    # Marker names from STO (remove _tx/_ty/_tz suffixes)
    time_col = df.columns[0]
    markers = [c[:-3] for c in df.columns[1::3]]

    # Frame and Time columns
    trc_header_row1 = "Frame#\tTime\t" + "\t\t\t".join(markers)
    trc_header_row2 = "\t\t" + "\t".join(
        [f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(n_markers)]
    )

    # --- Write TRC ---
    trc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trc_path, "w") as out:
        out.write(f"{header1}\n{header2}\n{header3}\n")
        out.write(f"{trc_header_row1}\n{trc_header_row2}\n")

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            time = row[time_col]
            coords = "\t".join(f"{x:.8f}" for x in row[1:])
            out.write(f"{i}\t{time:.6f}\t{coords}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert OpenSim STO marker file to TRC format."
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Path to the input STO file from OpenSim marker simulation.",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to the output TRC file.",
        required=True,
    )
    parser.add_argument(
        "--data-rate",
        type=float,
        default=50.0,
        help="Data sampling rate (Hz).",
    )
    parser.add_argument(
        "--camera-rate",
        type=float,
        default=50.0,
        help="Camera recording rate (Hz).",
    )
    parser.add_argument(
        "--units",
        type=str,
        default="m",
    )
    args = parser.parse_args()

    # Read CLI args.
    sto_path = Path(args.input).resolve()
    trc_path = Path(args.output).resolve()

    # Convert STO to TRC.
    sto_to_trc(
        sto_path,
        trc_path,
        data_rate=args.data_rate,
        camera_rate=args.camera_rate,
        units=args.units,
    )
