import json
import os
import warnings
from pathlib import Path
from typing import List

import cv2
import h5py
import numpy as np
import toml
from tqdm import tqdm

SUBJECTS = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]

MARKER_NAMES = [
    "Hip",
    "Spine_01",
    "Spine_02",
    "Spine_03",
    "Spine_04",
    "Spine_05",
    "Neck",
    "Neck_02",
    "Neck_03",
    "Head",
    "Nose",
    "REye",
    "LEye",
    "REar",
    "LEar",
    "RLatissimus",
    "LLatissimus",
    "RClavicle",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LClavicle",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "RBigToe",
    "RSmallToe",
    "RHeel",
    "LHip",
    "LKnee",
    "LAnkle",
    "LBigToe",
    "LSmallToe",
    "LHeel",
]

KINEMATIC_AXES = [
    "pelvis_tilt",
    "pelvis_list",
    "pelvis_rotation",
    "pelvis_tx",
    "pelvis_ty",
    "pelvis_tz",
    "hip_flexion_r",
    "hip_adduction_r",
    "hip_rotation_r",
    "knee_angle_r",
    "knee_angle_r_beta",
    "ankle_angle_r",
    "subtalar_angle_r",
    "mtp_angle_r",
    "hip_flexion_l",
    "hip_adduction_l",
    "hip_rotation_l",
    "knee_angle_l",
    "knee_angle_l_beta",
    "ankle_angle_l",
    "subtalar_angle_l",
    "mtp_angle_l",
    "L5_S1_Flex_Ext",
    "L5_S1_Lat_Bending",
    "L5_S1_axial_rotation",
    "L4_L5_Flex_Ext",
    "L4_L5_Lat_Bending",
    "L4_L5_axial_rotation",
    "L3_L4_Flex_Ext",
    "L3_L4_Lat_Bending",
    "L3_L4_axial_rotation",
    "L2_L3_Flex_Ext",
    "L2_L3_Lat_Bending",
    "L2_L3_axial_rotation",
    "L1_L2_Flex_Ext",
    "L1_L2_Lat_Bending",
    "L1_L2_axial_rotation",
    "L1_T12_Flex_Ext",
    "L1_T12_Lat_Bending",
    "L1_T12_axial_rotation",
    "Abs_r3",
    "Abs_r2",
    "Abs_r1",
    "Abs_t1",
    "Abs_t2",
    "neck_flexion",
    "neck_bending",
    "neck_rotation",
    "arm_flex_r",
    "arm_add_r",
    "arm_rot_r",
    "elbow_flex_r",
    "pro_sup_r",
    "wrist_flex_r",
    "wrist_dev_r",
    "arm_flex_l",
    "arm_add_l",
    "arm_rot_l",
    "elbow_flex_l",
    "pro_sup_l",
    "wrist_flex_l",
    "wrist_dev_l",
]


def build_index(root_dir: str, subjects: List[str] = SUBJECTS):
    index = {}
    skipped = []

    # Scan directory for TRC and MOT files
    for root, _, files in tqdm(os.walk(root_dir), "Scanning dataset files"):
        for file in files:
            if not file.startswith("S"):
                continue

            subject = file.split("_")[0]
            if subject not in subjects or subject in skipped:
                continue

            if subject not in index:
                # Find calibration file
                calibration = os.path.join(root_dir, "cameras", f"Calib_{subject}.toml")
                if not os.path.isfile(calibration):
                    warnings.warn(
                        f"Calibration file not found for subject {subject}: {calibration}. Skipping."
                    )
                    skipped.append(subject)
                    continue

                # Find OpenSim model file
                model_file = os.path.join(root_dir, "models", f"{subject}.osim")
                if not os.path.isfile(model_file):
                    warnings.warn(
                        f"OpenSim model file not found for subject {subject}: {model_file}. Skipping."
                    )
                    skipped.append(subject)
                    continue

                # Build subject entry
                index[subject] = dict(
                    calibration=calibration,
                    model_file=model_file,
                    data={},
                )

            # Determine action name and file type
            ext = os.path.splitext(file)[1]
            action_name = file[len(subject) + 1 : file.rfind(ext)]
            if action_name not in index[subject]["data"]:
                index[subject]["data"][action_name] = {}

            # Store file path
            fp = os.path.join(root, file)
            if ext == ".trc":
                index[subject]["data"][action_name]["markers"] = fp
            elif ext == ".mot":
                index[subject]["data"][action_name]["kinematics"] = fp
            else:
                warnings.warn(f"Encountered unknown file type: {fp}. Skipping.")

    # Filter out incomplete subjects
    # i.e., only keep actions that have both positions and kinematics, and only
    # keep subjects that have at least one complete action
    filtered_index = {}
    for subject, data in index.items():
        filtered_actions = {
            action: paths
            for action, paths in data["data"].items()
            if "markers" in paths and "kinematics" in paths
        }
        if filtered_actions:
            filtered_index[subject] = {
                "calibration": data["calibration"],
                "model_file": data["model_file"],
                "data": filtered_actions,
            }
    return dict(sorted(filtered_index.items()))


def read_trc(trc_path: str):
    """Reads a TRC file into numpy arrays (time and 3D positions).

    Returns:
        timestamps (np.ndarray): Time values [T].
        positions (np.ndarray): 3D marker positions [T, K, 3].
    """
    with open(trc_path, "r") as f:
        lines = f.readlines()

    # Read header
    header_line = 3
    data_start = 5
    header_parts = lines[header_line].strip().split("\t")

    # Check markers
    K = len(MARKER_NAMES)
    markers = [p for p in header_parts if p not in ["Frame#", "Time", ""]]
    indices = [markers.index(m) for m in MARKER_NAMES if m in markers]
    if len(indices) != K:
        raise ValueError(
            f"Markers in TRC file {trc_path} do not match expected MARKERS."
        )

    # Load data
    data = np.loadtxt(trc_path, skiprows=data_start)
    num_cols = K * 3 + 2
    if data.shape[1] != num_cols:
        raise ValueError(
            f"Unexpected number of columns in TRC file {trc_path}: "
            f"expected {num_cols}, got {data.shape[1]}"
        )

    # Reshape to [T, K, 3]
    T = data.shape[0]
    timestamps = data[:, 1]
    positions = data[:, 2:].reshape(T, len(markers), 3)
    positions = positions[:, indices, :]

    return timestamps, positions


def read_mot(mot_path: str):
    """Reads a MOT file into numpy arrays (time and data).

    Returns:
        timestamps (np.ndarray): Time values [T].
        data (np.ndarray): All motion parameters [T, D].
        columns (List[str]): Column names from header.
    """
    with open(mot_path, "r") as f:
        lines = f.readlines()

    # Find header end
    header_end = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_end = i
            break
    if header_end is None:
        raise ValueError(f"Missing 'endheader' in MOT file: {mot_path}")

    # Column names follow after header_end
    columns = lines[header_end + 1].strip().split("\t")
    if len(columns) < 2:
        raise ValueError(f"Invalid column header in MOT file: {mot_path}")

    # Load data
    data = np.loadtxt(mot_path, skiprows=header_end + 2)
    if data.shape[1] != len(columns):
        warnings.warn(
            f"Column count mismatch in MOT file {mot_path}: header={len(columns)}, data={data.shape[1]}"
        )

    timestamps = data[:, 0]
    values = data[:, 1:]

    axes_names = columns[1:]
    if axes_names != KINEMATIC_AXES:
        raise ValueError(
            f"Kinematic axes in MOT file {mot_path} do not match expected KINEMATIC_AXES."
        )

    return timestamps, values, axes_names


def read_calib(calib_path: str):
    """Load all camera calibration parameters from TOML into structured arrays."""
    calib = toml.load(calib_path)
    cam_names, K_list, dist_list, R_list, t_list, size_list = [], [], [], [], [], []

    for name, c in calib.items():
        if all(
            k in c for k in ["matrix", "distortions", "rotation", "translation", "size"]
        ):
            cam_names.append(name)
            K_list.append(np.array(c["matrix"], dtype=np.float32))
            dist_list.append(np.array(c["distortions"], dtype=np.float32))
            # Convert Rodrigues rotation vector to full rotation matrix
            rvec = np.array(c["rotation"], dtype=np.float32)
            R_list.append(cv2.Rodrigues(rvec)[0])
            t_list.append(np.array(c["translation"], dtype=np.float32).reshape(3))
            size_list.append(np.array(c["size"], dtype=np.int32))

    return dict(
        names=np.array(cam_names, dtype="S"),
        K=np.stack(K_list),
        dist=np.stack(dist_list),
        R=np.stack(R_list),
        t=np.stack(t_list),
        size=np.stack(size_list),
    )


def serialize_calibration(h5_group, calib_path: str):
    cams = read_calib(calib_path)
    calib_group = h5_group.create_group("calibration")
    for key, val in cams.items():
        calib_group.create_dataset(key, data=val, compression="gzip")


def serialize_to_hdf5(index: dict, output_file: str):
    with h5py.File(output_file, "w") as f:
        meta = f.create_group("metadata")
        meta.create_dataset("markers_names", data=np.array(MARKER_NAMES, dtype="S"))
        meta.create_dataset("kinematic_axes", data=np.array(KINEMATIC_AXES, dtype="S"))

        for subject, sdata in index.items():
            subj_group = f.create_group(subject)
            serialize_calibration(subj_group, sdata["calibration"])
            subj_group.attrs["model_file"] = sdata["model_file"]

            for action, paths in sdata["data"].items():
                act_group = subj_group.create_group(action)
                trc_file = paths["markers"]
                mot_file = paths["kinematics"]

                t_trc, positions = read_trc(trc_file)
                t_mot, kinematics, _ = read_mot(mot_file)

                # Synchronize frame counts
                min_len = min(len(t_trc), len(t_mot))
                positions, kinematics = positions[:min_len], kinematics[:min_len]
                timestamps = t_trc[:min_len]

                act_group.create_dataset("markers", data=positions, compression="gzip")
                act_group.create_dataset(
                    "kinematics", data=kinematics, compression="gzip"
                )
                act_group.create_dataset("timestamps", data=timestamps)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Serialize SimSpine dataset to HDF5")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing HDF5 file if it exists",
    )
    args = parser.parse_args()

    original_dir = Path("data/simspine/original")
    serialized_dir = Path("data/simspine/serialized")

    # Index the original dataset once
    # (this can be reused for future serializations)
    index_file = serialized_dir / "index.json"
    if index_file.exists():
        with open(index_file, "r") as f:
            index = json.load(f)
    else:
        print("Building dataset index...")
        index = build_index(original_dir)
        with open(index_file, "w") as f:
            json.dump(index, f, indent=4)

    h5_path = serialized_dir / "simspine.h5"
    if h5_path.exists() and not args.overwrite:
        print("HDF5 file already exists. Use --overwrite to replace it.")
    else:
        print("Serializing dataset to HDF5...")
        serialize_to_hdf5(index, output_file=h5_path)
        print("Serialization complete.")
