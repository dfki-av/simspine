from pathlib import Path

import numpy as np

MARKERS = [
    "Hip",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "Spine",
    "Neck",
    "Nose",
    # "Head",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RShoulder",
    "RElbow",
    "RWrist",
]


def read_trc(trc_path):
    """Reads a TRC file into numpy array [T, K, 3]."""
    with open(trc_path, "r") as f:
        lines = f.readlines()

    header_line = 3
    data_start = 5
    header_parts = lines[header_line].strip().split("\t")
    markers = [p for p in header_parts if p not in ["Frame#", "Time", ""]]

    data = np.loadtxt(trc_path, skiprows=data_start).copy()
    coords = data[:, 2:].reshape(len(data), len(markers), 3)
    return coords, markers


def _merge_trcs(pred, gt, markers):
    # Match only relevant markers
    keep = []
    for m in MARKERS:
        if m == "Spine":
            idx = (
                markers.index("Spine_04")
                if "Spine_04" in markers
                else markers.index("Spine")
            )
        elif m == "Neck":
            idx = (
                markers.index("Neck_01")
                if "Neck_01" in markers
                else markers.index("Neck")
            )
        else:
            idx = markers.index(m)
        keep.append(idx)

    T = min(pred.shape[0], gt.shape[0])
    pred[:T, keep, :] = gt[:T, :, [1, 2, 0]]
    return pred


def _build_trc_header(
    sequence_name: str,
    num_frames: int,
    num_markers: int,
    data_rate: float,
    camera_rate: float,
    markers: list,
) -> str:
    """
    TRC header format compatible with OpenSim.
    Units are m. Frame# starts at 1. Time starts at 0.0 with step 1/data_rate.
    """
    # Header rows
    hdr1 = f"PathFileType\t4\t(X/Y/Z)\t{sequence_name}.trc"
    hdr2 = "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
    # For compatibility, OrigDataRate is set to CameraRate and OrigNumFrames back-computed.
    orig_num_frames = int(round(num_frames * (camera_rate / data_rate)))
    hdr3 = f"{data_rate}\t{camera_rate}\t{num_frames}\t{num_markers}\tm\t{camera_rate}\t1\t{orig_num_frames}"

    # Row 4: marker labels (each with 2 empty tabs as in common TRC variants)
    labels_line = "Frame#\tTime\t" + "\t".join([f"{m}\t\t" for m in markers])
    # Row 5: coordinate labels X1 Y1 Z1 X2 Y2 Z2 ...
    coord_labels = "\t\t" + "\t".join(
        [f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(num_markers)]
    )

    return "\n".join([hdr1, hdr2, hdr3, labels_line, coord_labels])


def write_trc(
    output_path: Path,
    poses_m: np.ndarray,
    data_rate: float,
    camera_rate: float,
    sequence_name: str,
    markers=MARKERS,
):
    """
    Write TRC file. Input poses in meters with shape (T, 17, 3).
    Applies coordinate transforms:
      - Invert Y
      - Swap Y and Z (so final ordering is X, Y, Z in TRC with Y-up)
    Converts to millimeters before writing.
    """
    # if poses_m.ndim != 3 or poses_m.shape[1:] != (17, 3):
    #     raise ValueError(f"Expected poses of shape (T, 17, 3), got {poses_m.shape}")

    # Copy to avoid in-place on shared memory arrays
    data = poses_m.astype(np.float64).copy()

    # Flatten to (T, 51) = (T, 17*3)
    T = data.shape[0]
    flat = data.reshape(T, -1)

    # Prepend Frame and Time columns
    frame_numbers = np.arange(1, T + 1, dtype=np.int64).reshape(-1, 1)
    time = (np.arange(T, dtype=np.float64) / float(data_rate)).reshape(-1, 1)
    trc_matrix = np.hstack([frame_numbers, time, flat])  # shape (T, 2 + 51)

    header = _build_trc_header(
        sequence_name,
        T,
        len(markers),
        float(data_rate),
        float(camera_rate),
        markers=markers,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(header + "\n")
        # First column integer, rest floats
        fmt = ["%d"] + ["%.6f"] * (trc_matrix.shape[1] - 1)
        np.savetxt(f, trc_matrix, delimiter="\t", fmt=fmt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert predicted TRC files to standard format."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input TRC file with GT to merge with predictions.",
        required=True,
    )
    parser.add_argument(
        "--pred",
        "-p",
        type=str,
        help="Predicted TRC file with pseudo-markers.",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output merged TRC file.",
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
    args = parser.parse_args()

    # Read CLI args.
    inp_trc = Path(args.input).resolve()
    pred_trc = Path(args.pred).resolve()
    out_trc = Path(args.output).resolve()

    # Read input TRCs
    coords_gt, _ = read_trc(inp_trc)
    coords_pd, markers = read_trc(pred_trc)

    # Merge TRCs
    coords_merged = _merge_trcs(
        coords_pd,
        coords_gt,
        markers,
    )

    # Write merged TRC
    write_trc(
        out_trc,
        coords_merged,
        data_rate=args.data_rate,
        camera_rate=args.camera_rate,
        sequence_name=out_trc.stem,
        markers=markers,
    )
    print(f"Wrote merged TRC to {out_trc}")
