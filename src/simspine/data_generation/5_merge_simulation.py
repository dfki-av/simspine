from pathlib import Path

import numpy as np

SPINE_CHAIN = [
    "Hip",  # bottom anchor
    "Spine_01",
    "Spine_02",
    "Spine_03",
    "Spine_04",
    "Spine_05",
    "Neck",  # top anchor
    "Neck_02",
    "Neck_03",
]

ADDITIONAL_MARKERS = [
    "Head",
    "LClavicle",
    "RClavicle",
    # "LLatissimus",
    # "RLatissimus",
    "LHeel",
    "RHeel",
    "LBigToe",
    "RBigToe",
    "LSmallToe",
    "RSmallToe",
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


def get_spine(coords, markers):
    spind_idx = []
    for m in SPINE_CHAIN:
        idx = markers.index(m)
        spind_idx.append(idx)
    return coords[:, spind_idx, :], spind_idx


def merge_spine(pred, osim):
    pred_coords, pred_markers = pred
    osim_coords, osim_markers = osim

    # Identify spines
    pred_spine, spine_idx = get_spine(pred_coords, pred_markers)
    osim_spine, _ = get_spine(osim_coords, osim_markers)

    T = min(pred_coords.shape[0], osim_spine.shape[0])
    pred_spine = pred_spine[:T].copy()
    osim_spine = osim_spine[:T].copy()

    # Find indices of Hip and Neck anchors
    hip_idx = SPINE_CHAIN.index("Hip")
    neck_idx = SPINE_CHAIN.index("Neck")

    # For each frame
    for t in range(T):
        # --- 1. Anchors and vectors ---
        hip_pred = pred_spine[t, hip_idx]
        neck_pred = pred_spine[t, neck_idx]
        vec_pred = neck_pred - hip_pred
        len_pred = np.linalg.norm(vec_pred)

        hip_osim = osim_spine[t, hip_idx]
        neck_osim = osim_spine[t, neck_idx]
        vec_osim = neck_osim - hip_osim
        len_osim = np.linalg.norm(vec_osim)

        if len_pred < 1e-6 or len_osim < 1e-6:
            continue

        # --- 2. Align Hip–Neck direction (flip if needed) ---
        # Check if osim axis points opposite to pred axis
        if np.dot(vec_pred, vec_osim) < 0:
            # Reverse osim spine so Hip→Neck direction matches pred
            osim_spine[t] = osim_spine[t][::-1]
            # Update references after flip
            hip_osim = osim_spine[t, hip_idx]
            neck_osim = osim_spine[t, neck_idx]
            vec_osim = neck_osim - hip_osim
            len_osim = np.linalg.norm(vec_osim)

        # --- 3. Normalize and preserve curvature ---
        osim_rel = osim_spine[t] - hip_osim
        osim_axis = vec_osim / len_osim
        osim_proj = np.dot(osim_rel, osim_axis)
        osim_norm = osim_proj / len_osim
        osim_perp = osim_rel - np.outer(osim_proj / len_osim, vec_osim)

        new_axis = vec_pred / len_pred
        new_spine = (
            hip_pred + np.outer(osim_norm, vec_pred) + osim_perp * (len_pred / len_osim)
        )

        # --- 4. Apply local stretch constraint ---
        # Ensure Hip–Spine_01 distance ≤ 2 × Spine_01–Spine_02 distance
        if len(SPINE_CHAIN) >= 3:
            i_hip = 0  # first in chain
            i_s1 = 1
            i_s2 = 2
            d1 = np.linalg.norm(new_spine[i_s1] - new_spine[i_hip])
            d2 = np.linalg.norm(new_spine[i_s2] - new_spine[i_s1])
            if d1 > 2 * d2 and d2 > 1e-8:
                # Compress the Hip–Spine_01 segment toward Hip
                direction = (new_spine[i_s1] - new_spine[i_hip]) / d1
                new_spine[i_s1] = new_spine[i_hip] + direction * (2 * d2)

        # --- 5. Simulate latissimus dorsi effect ---
        # Get midpoint between Spine_04 and Spine_05
        # Then, get direction vectors from Spine_05 to both shoulders
        # Define latissimus points with same direction vector but starting at midpoint
        # instead of Spine_05, and length scaled by 0.6
        if "LShoulder" in pred_markers and "RShoulder" in pred_markers:
            i_s4 = SPINE_CHAIN.index("Spine_04")
            i_s5 = SPINE_CHAIN.index("Spine_05")
            i_shL = pred_markers.index("LShoulder")
            i_shR = pred_markers.index("RShoulder")
            mid_spine = 0.5 * (new_spine[i_s4] + new_spine[i_s5])
            dir_L = pred_coords[t, i_shL] - new_spine[i_s5]
            dir_R = pred_coords[t, i_shR] - new_spine[i_s5]
            len_L = np.linalg.norm(dir_L)
            len_R = np.linalg.norm(dir_R)
            if len_L > 1e-6:
                dir_L /= len_L
                latissimus_L = mid_spine + dir_L * (0.6 * len_L)
                i_latL = pred_markers.index("LLatissimus")
                pred_coords[t, i_latL] = latissimus_L
            if len_R > 1e-6:
                dir_R /= len_R
                latissimus_R = mid_spine + dir_R * (0.6 * len_R)
                i_latR = pred_markers.index("RLatissimus")
                pred_coords[t, i_latR] = latissimus_R

        # Smooth predicted spine using a Laplacian filter
        smoothed_spine = pred_spine.copy()
        for i in range(1, len(SPINE_CHAIN) - 1):
            smoothed_spine[t, i] = 0.5 * (
                smoothed_spine[t, i - 1] + smoothed_spine[t, i + 1]
            )

        # --- 6. Blend spine ---
        blend_factor = (
            0.8  # Adjust blending factor as needed (0.0 = only pred, 1.0 = only osim)
        )
        pred_coords[t, spine_idx, :] = (
            blend_factor * new_spine + (1 - blend_factor) * smoothed_spine[t]
        )

        # replace additional markers if present
        for m in ADDITIONAL_MARKERS:
            i_m = osim_markers.index(m)
            i_m_pred = pred_markers.index(m)
            pred_coords[t, i_m_pred] = osim_coords[t, i_m]

    return pred_coords


def build_trc_header(
    sequence_name: str,
    num_frames: int,
    num_markers: int,
    data_rate: float,
    camera_rate: float,
    markers: list,
) -> str:
    """
    TRC header format compatible with OpenSim.
    Units are mm. Frame# starts at 1. Time starts at 0.0 with step 1/data_rate.
    """
    # Header rows
    hdr1 = f"PathFileType\t4\t(X/Y/Z)\t{sequence_name}.trc"
    hdr2 = "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
    # For compatibility, OrigDataRate is set to CameraRate and OrigNumFrames back-computed.
    orig_num_frames = int(round(num_frames * (camera_rate / data_rate)))
    hdr3 = f"{data_rate}\t{camera_rate}\t{num_frames}\t{num_markers}\tmm\t{camera_rate}\t1\t{orig_num_frames}"

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
    markers=SPINE_CHAIN,
):
    """
    Write TRC file. Input poses in meters with shape (T, 17, 3).
    Applies coordinate transforms:
      - Invert Y
      - Swap Y and Z (so final ordering is X, Y, Z in TRC with Y-up)
    Converts to millimeters before writing.
    """
    # Copy to avoid in-place on shared memory arrays
    data = poses_m.astype(np.float64).copy()

    # Flatten to (T, 51) = (T, 17*3)
    T = data.shape[0]
    flat = data.reshape(T, -1)

    # Prepend Frame and Time columns
    frame_numbers = np.arange(1, T + 1, dtype=np.int64).reshape(-1, 1)
    time = (np.arange(T, dtype=np.float64) / float(data_rate)).reshape(-1, 1)
    trc_matrix = np.hstack([frame_numbers, time, flat])  # shape (T, 2 + 51)

    header = build_trc_header(
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
        help="Path to input TRC file.",
        required=True,
    )
    parser.add_argument(
        "--simulated",
        "-s",
        type=str,
        help="Path to simulated TRC file.",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to output TRC file.",
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
    sim_trc = Path(args.simulated).resolve()
    out_trc = Path(args.output).resolve()

    coords, markers = read_trc(inp_trc)
    coords_gt, markers_gt = read_trc(sim_trc)
    coords = merge_spine(
        (coords, markers),
        (coords_gt, markers_gt),
    )

    # Write to output TRC
    write_trc(
        out_trc,
        coords,
        data_rate=args.data_rate,
        camera_rate=args.camera_rate,
        sequence_name=out_trc.stem,
        markers=markers,
    )
