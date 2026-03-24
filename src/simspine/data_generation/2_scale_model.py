import argparse
from pathlib import Path

import numpy as np
import opensim
import pandas as pd
from anytree import PreOrderIter
from lxml import etree

OPENSIM_SETUP_DIR = Path("assets/OpenSim_Setup")
MODEL_PATH = OPENSIM_SETUP_DIR / "Model.osim"
MARKERS_PATH = OPENSIM_SETUP_DIR / "Markers.xml"
IK_SETUP_PATH = OPENSIM_SETUP_DIR / "IK_Setup.xml"
SCALING_SETUP_PATH = OPENSIM_SETUP_DIR / "Scaling_Setup.xml"

angle_dict = {  # lowercase!
    # joint angles
    "right ankle": [["RKnee", "RAnkle", "RBigToe", "RHeel"], "dorsiflexion", 90, 1],
    "left ankle": [["LKnee", "LAnkle", "LBigToe", "LHeel"], "dorsiflexion", 90, 1],
    "right knee": [["RAnkle", "RKnee", "RHip"], "flexion", -180, 1],
    "left knee": [["LAnkle", "LKnee", "LHip"], "flexion", -180, 1],
    "right hip": [["RKnee", "RHip", "Hip", "Neck"], "flexion", 0, -1],
    "left hip": [["LKnee", "LHip", "Hip", "Neck"], "flexion", 0, -1],
    # 'lumbar': [['Neck', 'Hip', 'RHip', 'LHip'], 'flexion', -180, -1],
    # 'neck': [['Head', 'Neck', 'RShoulder', 'LShoulder'], 'flexion', -180, -1],
    "right shoulder": [["RElbow", "RShoulder", "Hip", "Neck"], "flexion", 0, -1],
    "left shoulder": [["LElbow", "LShoulder", "Hip", "Neck"], "flexion", 0, -1],
    "right elbow": [["RWrist", "RElbow", "RShoulder"], "flexion", 180, -1],
    "left elbow": [["LWrist", "LElbow", "LShoulder"], "flexion", 180, -1],
    "right wrist": [["RElbow", "RWrist", "RIndex"], "flexion", -180, 1],
    "left wrist": [["LElbow", "LIndex", "LWrist"], "flexion", -180, 1],
    # segment angles
    "right foot": [["RBigToe", "RHeel"], "horizontal", 0, -1],
    "left foot": [["LBigToe", "LHeel"], "horizontal", 0, -1],
    "right shank": [["RAnkle", "RKnee"], "horizontal", 0, -1],
    "left shank": [["LAnkle", "LKnee"], "horizontal", 0, -1],
    "right thigh": [["RKnee", "RHip"], "horizontal", 0, -1],
    "left thigh": [["LKnee", "LHip"], "horizontal", 0, -1],
    "pelvis": [["LHip", "RHip"], "horizontal", 0, -1],
    "trunk": [["Neck", "Hip"], "horizontal", 0, -1],
    "shoulders": [["LShoulder", "RShoulder"], "horizontal", 0, -1],
    "head": [["Head", "Neck"], "horizontal", 0, -1],
    "right arm": [["RElbow", "RShoulder"], "horizontal", 0, -1],
    "left arm": [["LElbow", "LShoulder"], "horizontal", 0, -1],
    "right forearm": [["RWrist", "RElbow"], "horizontal", 0, -1],
    "left forearm": [["LWrist", "LElbow"], "horizontal", 0, -1],
    "right hand": [["RIndex", "RWrist"], "horizontal", 0, -1],
    "left hand": [["LIndex", "LWrist"], "horizontal", 0, -1],
}


def points2D_to_angles(points_list):
    """
    If len(points_list)==2, computes clockwise angle of ab vector w.r.t. horizontal (e.g. RBigToe, RHeel)
    If len(points_list)==3, computes clockwise angle from a to c around b (e.g. Neck, Hip, Knee)
    If len(points_list)==4, computes clockwise angle between vectors ab and cd (e.g. Neck Hip, RKnee RHip)

    If parameters are float, returns a float between 0.0 and 360.0
    If parameters are arrays, returns an array of floats between 0.0 and 360.0
    """

    if len(points_list) < 2:  # if not enough points, return None
        return np.nan

    ax, ay = points_list[0]
    bx, by = points_list[1]

    if len(points_list) == 2:
        ux, uy = ax - bx, ay - by
        vx, vy = 1, 0
    if len(points_list) == 3:
        cx, cy = points_list[2]
        ux, uy = ax - bx, ay - by
        vx, vy = cx - bx, cy - by

    if len(points_list) == 4:
        cx, cy = points_list[2]
        dx, dy = points_list[3]
        ux, uy = bx - ax, by - ay
        vx, vy = dx - cx, dy - cy

    ang = np.arctan2(uy, ux) - np.arctan2(vy, vx)
    return np.degrees(ang)

    # ang_deg = np.array(np.degrees(np.unwrap(ang*2)/2))
    # return ang_deg


def points_to_angles(points_list):
    """
    If len(points_list)==2, computes clockwise angle of ab vector w.r.t. horizontal (e.g. RBigToe, RHeel)
    If len(points_list)==3, computes clockwise angle from a to c around b (e.g. Neck, Hip, Knee)
    If len(points_list)==4, computes clockwise angle between vectors ab and cd (e.g. Neck Hip, RKnee RHip)

    Points can be 2D or 3D.
    If parameters are float, returns a float between 0.0 and 360.0
    If parameters are arrays, returns an array of floats between 0.0 and 360.0

    INPUTS:
    - points_list: list of arrays of points

    OUTPUTS:
    - ang_deg: float or array of floats. The angle(s) in degrees.
    """

    if len(points_list) < 2:  # if not enough points, return None
        return np.nan

    points_array = np.array(points_list)
    dimensions = points_array.shape[-1]

    if len(points_list) == 2:
        vector_u = points_array[0] - points_array[1]
        if len(points_array.shape) == 2:
            vector_v = np.array(
                [1, 0, 0]
            )  # Here vector X, could be any horizontal vector
        else:
            vector_v = np.array(
                [
                    [1, 0, 0],
                ]
                * points_array.shape[1]
            )

    elif len(points_list) == 3:
        vector_u = points_array[0] - points_array[1]
        vector_v = points_array[2] - points_array[1]

    elif len(points_list) == 4:
        vector_u = points_array[1] - points_array[0]
        vector_v = points_array[3] - points_array[2]

    else:
        return np.nan

    if dimensions == 2:
        vector_u = vector_u[:2]
        vector_v = vector_v[:2]
        ang = np.arctan2(vector_u[1], vector_u[0]) - np.arctan2(
            vector_v[1], vector_v[0]
        )
    else:
        cross_product = np.cross(vector_u, vector_v)
        dot_product = np.einsum(
            "ij,ij->i", vector_u, vector_v
        )  # np.dot(vector_u, vector_v) # does not work with time series
        ang = np.arctan2(np.linalg.norm(cross_product, axis=1), dot_product)

    return np.degrees(ang)
    # ang_deg = np.array(np.degrees(np.unwrap(ang*2)/2))


def fixed_angles(points_list, ang_name):
    """
    Add offset and multiplying factor to angles

    INPUTS:
    - points_list: list of arrays of points
    - ang_name: str. The name of the angle to consider.

    OUTPUTS:
    - ang: float. The angle in degrees.
    """

    ang_params = angle_dict[ang_name]
    ang = points_to_angles(points_list)
    ang += ang_params[2]
    ang *= ang_params[3]
    if ang_name in ["pelvis", "shoulders"]:
        ang = np.where(ang > 90, ang - 180, ang)
        ang = np.where(ang < -90, ang + 180, ang)
    else:
        ang = np.where(ang > 180, ang - 360, ang)
        ang = np.where(ang < -180, ang + 360, ang)

    return ang


def mean_angles(
    Q_coords, ang_to_consider=["right knee", "left knee", "right hip", "left hip"]
):
    """
    Compute the mean angle time series from 3D points for a given list of angles.

    INPUTS:
    - Q_coords (DataFrame): The triangulated coordinates of the markers.
    - ang_to_consider (list): The list of angles to consider (requires angle_dict).

    OUTPUTS:
    - ang_mean: The mean angle time series.
    """

    ang_to_consider = ["right knee", "left knee", "right hip", "left hip"]

    angs = []
    for ang_name in ang_to_consider:
        ang_params = angle_dict[ang_name]
        ang_mk = ang_params[0]
        if "Neck" not in Q_coords.columns:
            df_MidShoulder = pd.DataFrame(
                (Q_coords["RShoulder"].values + Q_coords["LShoulder"].values) / 2
            )
            df_MidShoulder.columns = ["Neck"] * 3
            Q_coords = pd.concat(
                (Q_coords.reset_index(drop=True), df_MidShoulder), axis=1
            )

        pts_for_angles = []
        for pt in ang_mk:
            # pts_for_angles.append(Q_coords.iloc[:,markers.index(pt)*3:markers.index(pt)*3+3])
            pts_for_angles.append(Q_coords[pt])

        ang = fixed_angles(pts_for_angles, ang_name)
        ang = np.abs(ang)
        angs.append(ang)

    return np.mean(angs, axis=0)


def natural_sort_key(s):
    """
    Sorts list of strings with numbers in natural order (alphabetical and numerical)
    Example: ['item_1', 'item_2', 'item_10', 'stuff_1']
    """
    s = str(s)
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


def best_coords_for_measurements(
    Q_coords,
    keypoints_names,
    fastest_frames_to_remove_percent=0.2,
    close_to_zero_speed=0.2,
    large_hip_knee_angles=45,
):
    """
    Compute the best coordinates for measurements, after removing:
    - 20% fastest frames (may be outliers)
    - frames when speed is close to zero (person is out of frame): 0.2 m/frame, or 50 px/frame
    - frames when hip and knee angle below 45° (imprecise coordinates when person is crouching)

    INPUTS:
    - Q_coords: pd.DataFrame. The XYZ coordinates of each marker
    - keypoints_names: list. The list of marker names
    - fastest_frames_to_remove_percent: float
    - close_to_zero_speed: float (sum for all keypoints: about 50 px/frame or 0.2 m/frame)
    - large_hip_knee_angles: int
    - trimmed_extrema_percent

    OUTPUT:
    - Q_coords_low_speeds_low_angles: pd.DataFrame. The best coordinates for measurements
    """

    # Add MidShoulder column
    df_MidShoulder = pd.DataFrame(
        (Q_coords["RShoulder"].values + Q_coords["LShoulder"].values) / 2
    )
    df_MidShoulder.columns = ["MidShoulder"] * 3
    Q_coords = pd.concat((Q_coords.reset_index(drop=True), df_MidShoulder), axis=1)

    # Add Hip column if not present
    n_markers_init = len(keypoints_names)
    if "Hip" not in keypoints_names:
        df_Hip = pd.DataFrame((Q_coords["RHip"].values + Q_coords["LHip"].values) / 2)
        df_Hip.columns = ["Hip"] * 3
        Q_coords = pd.concat((Q_coords.reset_index(drop=True), df_Hip), axis=1)
    n_markers = len(keypoints_names)

    # Using 80% slowest frames
    sum_speeds = pd.Series(
        np.nansum(
            [
                np.linalg.norm(Q_coords.iloc[:, kpt : kpt + 3].diff(), axis=1)
                for kpt in range(n_markers)
            ],
            axis=0,
        )
    )
    sum_speeds = sum_speeds[
        sum_speeds > close_to_zero_speed
    ]  # Removing when speeds close to zero (out of frame)
    if len(sum_speeds) == 0:
        print(
            "All frames have speed close to zero. Make sure the person is moving and correctly detected, or change close_to_zero_speed to a lower value. Not restricting the speeds to be above any threshold."
        )
        Q_coords_low_speeds = Q_coords
    else:
        min_speed_indices = (
            sum_speeds.abs()
            .nsmallest(int(len(sum_speeds) * (1 - fastest_frames_to_remove_percent)))
            .index
        )
        Q_coords_low_speeds = Q_coords.iloc[min_speed_indices].reset_index(drop=True)

    # Only keep frames with hip and knee flexion angles below 45%
    # (if more than 50 of them, else take 50 smallest values)
    try:
        ang_mean = mean_angles(
            Q_coords_low_speeds,
            ang_to_consider=["right knee", "left knee", "right hip", "left hip"],
        )
        Q_coords_low_speeds_low_angles = Q_coords_low_speeds[
            ang_mean < large_hip_knee_angles
        ]
        if len(Q_coords_low_speeds_low_angles) < 50:
            Q_coords_low_speeds_low_angles = Q_coords_low_speeds.iloc[
                pd.Series(ang_mean).nsmallest(50).index
            ]
    except:
        Q_coords_low_speeds_low_angles = Q_coords_low_speeds
        print(
            f"At least one among the RAnkle, RKnee, RHip, RShoulder, LAnkle, LKnee, LHip, LShoulder markers is missing for computing the knee and hip angles. Not restricting these angles to be below {large_hip_knee_angles}°."
        )

    if n_markers_init < n_markers:
        Q_coords_low_speeds_low_angles = Q_coords_low_speeds_low_angles.iloc[:, :-3]

    return Q_coords_low_speeds_low_angles


def compute_height(
    Q_coords,
    keypoints_names,
    fastest_frames_to_remove_percent=0.1,
    close_to_zero_speed=50,
    large_hip_knee_angles=45,
    trimmed_extrema_percent=0.5,
):
    """
    Compute the height of the person from the trc data.

    INPUTS:
    - Q_coords: pd.DataFrame. The XYZ coordinates of each marker
    - keypoints_names: list. The list of marker names
    - fastest_frames_to_remove_percent: float. Frames with high speed are considered as outliers
    - close_to_zero_speed: float. Sum for all keypoints: about 50 px/frame or 0.2 m/frame
    - large_hip_knee_angles5: float. Hip and knee angles below this value are considered as imprecise
    - trimmed_extrema_percent: float. Proportion of the most extreme segment values to remove before calculating their mean)

    OUTPUT:
    - height: float. The estimated height of the person
    """

    # Retrieve most reliable coordinates, adding MidShoulder and Hip columns if not present
    Q_coords_low_speeds_low_angles = best_coords_for_measurements(
        Q_coords,
        keypoints_names,
        fastest_frames_to_remove_percent=fastest_frames_to_remove_percent,
        close_to_zero_speed=close_to_zero_speed,
        large_hip_knee_angles=large_hip_knee_angles,
    )

    # Automatically compute the height of the person
    try:
        feet_pairs = [["RHeel", "RAnkle"], ["LHeel", "LAnkle"]]
        rfoot, lfoot = [
            euclidean_distance(
                Q_coords_low_speeds_low_angles[pair[0]],
                Q_coords_low_speeds_low_angles[pair[1]],
            )
            for pair in feet_pairs
        ]
    except:
        rfoot, lfoot = 0.10, 0.10
        print(
            "The Heel marker is missing from your model. Considering Foot to Heel size as 10 cm."
        )

    ankle_to_shoulder_pairs = [
        ["RAnkle", "RKnee"],
        ["RKnee", "RHip"],
        ["RHip", "RShoulder"],
        ["LAnkle", "LKnee"],
        ["LKnee", "LHip"],
        ["LHip", "LShoulder"],
    ]
    try:
        rshank, rfemur, rback, lshank, lfemur, lback = [
            euclidean_distance(
                Q_coords_low_speeds_low_angles[pair[0]],
                Q_coords_low_speeds_low_angles[pair[1]],
            )
            for pair in ankle_to_shoulder_pairs
        ]
    except:
        print(
            'At least one of the following markers is missing for computing the height of the person:\
                            RAnkle, RKnee, RHip, RShoulder, LAnkle, LKnee, LHip, LShoulder.\n\
                            Make sure that the person is entirely visible, or use a calibration file instead, or set "to_meters=false".'
        )
        raise ValueError(
            'At least one of the following markers is missing for computing the height of the person:\
                         RAnkle, RKnee, RHip, RShoulder, LAnkle, LKnee, LHip, LShoulder.\
                         Make sure that the person is entirely visible, or use a calibration file instead, or set "to_meters=false".'
        )

    try:
        head_pair = [["MidShoulder", "Head"]]
        head = [
            euclidean_distance(
                Q_coords_low_speeds_low_angles[pair[0]].iloc[:, :3],
                Q_coords_low_speeds_low_angles[pair[1]].iloc[:, :3],
            )
            for pair in head_pair
        ][0]
    except:
        print(
            "The Head marker is missing from your model. Considering Neck to Head size as 1.33 times Neck to MidShoulder size."
        )
        head_pair = [["MidShoulder", "Nose"]]
        head = [
            euclidean_distance(
                Q_coords_low_speeds_low_angles[pair[0]].iloc[:, :3],
                Q_coords_low_speeds_low_angles[pair[1]].iloc[:, :3],
            )
            for pair in head_pair
        ][0] * 1.33

    heights = (
        (rfoot + lfoot) / 2
        + (rshank + lshank) / 2
        + (rfemur + lfemur) / 2
        + (rback + lback) / 2
        + head
    )

    # Remove the 20% most extreme values
    return trimmed_mean(heights, trimmed_extrema_percent=trimmed_extrema_percent)


def read_trc(trc_path):
    """
    Read a TRC file and extract its contents.

    INPUTS:
    - trc_path (str): The path to the TRC file.

    OUTPUTS:
    - tuple: A tuple containing the Q coordinates, frames column, time column, marker names, and header.
    """

    try:
        with open(trc_path, "r") as trc_file:
            header = [next(trc_file) for _ in range(5)]
        markers = header[3].split("\t")[2::3]
        markers = [m.strip() for m in markers if m.strip()]  # remove last \n character

        trc_df = pd.read_csv(trc_path, sep="\t", skiprows=4, encoding="utf-8")
        frames_col, time_col = trc_df.iloc[:, 0], trc_df.iloc[:, 1]
        Q_coords = trc_df.drop(trc_df.columns[[0, 1]], axis=1)
        Q_coords = Q_coords.loc[
            :, ~Q_coords.columns.str.startswith("Unnamed")
        ]  # remove unnamed columns
        Q_coords.columns = np.array([[m, m, m] for m in markers]).ravel().tolist()

        return Q_coords, frames_col, time_col, markers, header

    except Exception as e:
        raise ValueError(f"Error reading TRC file at {trc_path}: {e}")


def trimmed_mean(arr, trimmed_extrema_percent=0.5):
    """
    Trimmed mean calculation for an array.

    INPUTS:
    - arr (np.array): The input array.
    - trimmed_extrema_percent (float): The percentage of values to be trimmed from both ends.

    OUTPUTS:
    - float: The trimmed mean of the array.
    """

    # Sort the array
    sorted_arr = np.sort(arr)

    # Determine the indices for the 25th and 75th percentiles (if trimmed_percent = 0.5)
    lower_idx = int(len(sorted_arr) * (trimmed_extrema_percent / 2))
    upper_idx = int(len(sorted_arr) * (1 - trimmed_extrema_percent / 2))

    # Slice the array to exclude the 25% lowest and highest values
    trimmed_arr = sorted_arr[lower_idx:upper_idx]

    # Return the mean of the remaining values
    return np.mean(trimmed_arr)


def euclidean_distance(q1, q2):
    """
    Euclidean distance between 2 points (N-dim).

    INPUTS:
    - q1: list of N_dimensional coordinates of point
         or list of N points of N_dimensional coordinates
    - q2: idem

    OUTPUTS:
    - euc_dist: float. Euclidian distance between q1 and q2
    """

    q1 = np.array(q1)
    q2 = np.array(q2)
    dist = q2 - q1
    if np.isnan(dist).all():
        dist = np.empty_like(dist)
        dist[...] = np.inf

    if len(dist.shape) == 1:
        euc_dist = np.sqrt(np.nansum([d**2 for d in dist]))
    else:
        euc_dist = np.sqrt(np.nansum([d**2 for d in dist], axis=1))

    return euc_dist


def get_kpt_pairs_from_tree(root_node):
    """
    Get marker pairs for all parent-child relationships in the tree.
    # Excludes the root node.
    # Not used in the current version.

    INPUTS:
    - root_node (Node): The root node of the tree.

    OUTPUTS:
    - list: A list of name pairs for all parent-child relationships in the tree.
    """

    pairs = []
    for node in PreOrderIter(root_node):
        # if node.is_root:
        #     continue
        for child in node.children:
            pairs.append([node.name, child.name])

    return pairs


def get_kpt_pairs_from_scaling(scaling_root):
    """
    Get all marker pairs from the scaling setup file.

    INPUTS:
    - scaling_root (Element): The root element of the scaling setup file.

    OUTPUTS:
    - pairs: A list of marker pairs.
    """

    return [
        pair.find("markers").text.strip().split(" ")
        for pair in scaling_root[0].findall(".//MarkerPair")
    ]


def dict_segment_marker_pairs(scaling_root):
    """
    Get a dictionary of segment names and their corresponding marker pairs.

    Args:
        scaling_root (Element): The root element of the scaling setup file.

    Returns:
        segment_markers_dict: A dictionary of segment names and their corresponding marker pairs.
    """

    segment_markers_dict = {}
    for measurement in scaling_root.findall(".//Measurement"):
        # Collect all marker pairs for this measurement
        marker_pairs = [
            pair.find("markers").text.strip().split()
            for pair in measurement.findall(".//MarkerPair")
        ]

        # Collect all body scales for this measurement
        for body_scale in measurement.findall(".//BodyScale"):
            body_name = body_scale.get("name")
            axes = body_scale.find("axes").text.strip().split()
            for axis in axes:
                body_name_axis = f"{body_name}_{axis}"

                # left/right symmetry
                segment_markers_dict.setdefault(body_name_axis, []).extend(marker_pairs)

    return segment_markers_dict


def dict_segment_ratio(
    scaling_root,
    unscaled_model,
    Q_coords_scaling,
    markers,
    trimmed_extrema_percent=0.5,
):
    """
    Calculate the ratios between the size of the actual segment and the size of the model segment.
    X, Y, and Z ratios are calculated separately if the original scaling setup file asks for it.

    Args:
        scaling_root (Element): The root element of the scaling setup file.
        unscaled_model (Model): The original OpenSim model before scaling.
        Q_coords_scaling (DataFrame): The triangulated coordinates of the markers.
        markers (list): The list of marker names.
        trimmed_extrema_percent (float): The proportion of the most extreme segment values to remove before calculating their mean.

    Returns:
        segment_ratio_dict: A dictionary of segment names and their corresponding X, Y, and Z ratios.
    """

    segment_pairs = get_kpt_pairs_from_scaling(scaling_root)

    # Get median segment lengths from Q_coords_scaling. Trimmed mean works better than mean or median
    trc_segment_lengths = np.array(
        [
            euclidean_distance(
                Q_coords_scaling.iloc[
                    :, markers.index(pt1) * 3 : markers.index(pt1) * 3 + 3
                ],
                Q_coords_scaling.iloc[
                    :, markers.index(pt2) * 3 : markers.index(pt2) * 3 + 3
                ],
            )
            for (pt1, pt2) in segment_pairs
        ]
    )
    # trc_segment_lengths = np.median(trc_segment_lengths, axis=1)
    # trc_segment_lengths = np.mean(trc_segment_lengths, axis=1)
    trc_segment_lengths = np.array(
        [
            trimmed_mean(arr, trimmed_extrema_percent=trimmed_extrema_percent)
            for arr in trc_segment_lengths
        ]
    )

    # Get model segment lengths
    model_markers = [
        marker
        for marker in markers
        if marker in [m.getName() for m in unscaled_model.getMarkerSet()]
    ]
    model_markers_locs = [
        unscaled_model.getMarkerSet()
        .get(marker)
        .getLocationInGround(unscaled_model.getWorkingState())
        .to_numpy()
        for marker in model_markers
    ]
    model_segment_lengths = np.array(
        [
            euclidean_distance(
                model_markers_locs[model_markers.index(pt1)],
                model_markers_locs[model_markers.index(pt2)],
            )
            for (pt1, pt2) in segment_pairs
        ]
    )

    # Calculate ratio for each segment
    segment_ratios = trc_segment_lengths / model_segment_lengths
    segment_markers_dict = dict_segment_marker_pairs(scaling_root)
    segment_ratio_dict_temp = segment_markers_dict.copy()
    segment_ratio_dict_temp.update(
        {
            key: np.mean(
                [
                    segment_ratios[segment_pairs.index(k)]
                    for k in segment_markers_dict[key]
                ]
            )
            for key in segment_markers_dict
        }
    )
    # Merge X, Y, Z ratios into single key
    segment_ratio_dict = {}
    xyz_keys = list({key[:-2] for key in segment_ratio_dict_temp})
    for key in xyz_keys:
        segment_ratio_dict[key] = [
            segment_ratio_dict_temp[key + "_X"],
            segment_ratio_dict_temp[key + "_Y"],
            segment_ratio_dict_temp[key + "_Z"],
        ]

    return segment_ratio_dict


def deactivate_measurements(scaling_root):
    """
    Deactivate all scalings based on marker positions (called 'measurements' in OpenSim) in the scaling setup file.
    (will use scaling based on segment sizes instead (called 'manual' in OpenSim))

    INPUTS:
    - scaling_root (Element): The root element of the scaling setup file.

    OUTPUTS:
    - scaling_root with deactivated measurements.
    """

    measurement_set = scaling_root.find(".//MeasurementSet/objects")
    for measurement in measurement_set.findall("Measurement"):
        apply_elem = measurement.find("apply")
        apply_elem.text = "false"


def update_scale_values(scaling_root, segment_ratio_dict):
    """
    Remove previous scaling values ('manual') and
    add new scaling values based on calculated segment ratios.

    INPUTS:
    - scaling_root (Element): The root element of the scaling setup file.
    - segment_ratio_dict (dict): A dictionary of segment names and their corresponding X, Y, and Z ratios.

    OUTPUTS:
    - scaling_root with updated scaling values.
    """

    # Get the ScaleSet/objects element
    scale_set = scaling_root.find(".//ScaleSet/objects")

    # Remove all existing Scale elements
    for scale in scale_set.findall("Scale"):
        scale_set.remove(scale)

    # Add new Scale elements based on scale_dict
    for segment, scales in segment_ratio_dict.items():
        new_scale = etree.Element("Scale")
        # scales
        scales_elem = etree.SubElement(new_scale, "scales")
        scales_elem.text = " ".join(map(str, scales))
        # segment name
        segment_elem = etree.SubElement(new_scale, "segment")
        segment_elem.text = segment
        # apply True
        apply_elem = etree.SubElement(new_scale, "apply")
        apply_elem.text = "true"

        scale_set.append(new_scale)


def aggregate_trc_frames(
    trc_folder,
    fastest_frames_to_remove_percent=0.1,
    close_to_zero_speed_m=0.2,
    large_hip_knee_angles=45,
):
    """Load and concatenate all TRC coordinate data."""
    all_coords = []
    for trc_file in sorted(Path(trc_folder).glob("*.trc")):
        Q, _, _, markers, _ = read_trc(trc_file)
        all_coords.append(Q)
    if not all_coords:
        raise FileNotFoundError(f"No TRC files found in {trc_folder}")
    Q_coords = pd.concat(all_coords, ignore_index=True)

    # Remove fastest frames, frames with null speed, and frames with large hip and knee angles
    Q_coords_low_speeds_low_angles = best_coords_for_measurements(
        Q_coords,
        markers,
        fastest_frames_to_remove_percent=fastest_frames_to_remove_percent,
        large_hip_knee_angles=large_hip_knee_angles,
        close_to_zero_speed=close_to_zero_speed_m,
    )

    if Q_coords_low_speeds_low_angles.size == 0:
        print(
            f"\nNo frames left after removing fastest frames, frames with null speed, and frames with large hip and knee angles for {trc_file}. The person may be static, or crouched, or incorrectly detected."
        )
        print(
            "Running with fastest_frames_to_remove_percent=0, close_to_zero_speed_m=0, large_hip_knee_angles=0, trimmed_extrema_percent=0. You can edit these parameters in your Config.toml file.\n"
        )
        Q_coords_low_speeds_low_angles = Q_coords

    return Q_coords_low_speeds_low_angles, markers


def perform_scaling(
    trc_folder: Path,
    scaled_model_path: Path,
    subject_height=1.75,
    subject_mass=70.0,
    fastest_frames_to_remove_percent=0.1,
    close_to_zero_speed_m=0.2,
    large_hip_knee_angles=45,
    trimmed_extrema_percent=0.5,
):
    """
    Perform model scaling based on the (not necessarily static) TRC file:
    - Remove 10% fastest frames (potential outliers)
    - Remove frames where coordinate speed is null (person probably out of frame)
    - Remove 40% most extreme calculated segment values (potential outliers)
    - For each segment, scale on the mean of the remaining segment values

    Args:
        trc_folder (Path): The path to the folder containing TRC files.
        scaled_model_path (Path): The path to save the scaled model.
        subject_height (float): The height of the subject.
        subject_mass (float): The mass of the subject.
        fastest_frames_to_remove_percent (float): Fasters frames may be outliers
        large_hip_knee_angles (float): Imprecise coordinates when person is crouching
        trimmed_extrema_percent (float): Proportion of the most extreme segment values to remove before calculating their mean
    """
    # Get subject name from scaled model path
    subject_name = scaled_model_path.stem
    print(f"Scaling model for subject: {subject_name} ...")

    output_dir = scaled_model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    opensim.ModelVisualizer.addDirToGeometrySearchPaths(
        str(OPENSIM_SETUP_DIR / "Geometry")
    )
    unscaled_model_path = MODEL_PATH
    if not unscaled_model_path:
        raise ValueError(f"Unscaled OpenSim model not found at: {unscaled_model_path}")
    unscaled_model = opensim.Model(str(unscaled_model_path))

    # Add markers to model
    markers_path = MARKERS_PATH
    markerset = opensim.MarkerSet(str(markers_path))
    unscaled_model.set_MarkerSet(markerset)

    # Initialize and save model with markers
    unscaled_model.initSystem()
    scaled_model_path = str(scaled_model_path.resolve())
    unscaled_model.printToXML(scaled_model_path)

    # Load scaling setup
    scaling_path = SCALING_SETUP_PATH
    scaling_tree = etree.parse(scaling_path)
    scaling_root = scaling_tree.getroot()
    scaling_path_temp = str(output_dir / (subject_name + "_scaling_setup.xml"))

    # Remove fastest frames, frames with null speed, and frames with large hip and knee angles
    Q_coords_low_speeds_low_angles, markers = aggregate_trc_frames(
        trc_folder,
        fastest_frames_to_remove_percent=fastest_frames_to_remove_percent,
        close_to_zero_speed_m=close_to_zero_speed_m,
        large_hip_knee_angles=large_hip_knee_angles,
    )

    # Get manual scale values (mean from remaining frames after trimming the 20% most extreme values)
    segment_ratio_dict = dict_segment_ratio(
        scaling_root,
        unscaled_model,
        Q_coords_low_speeds_low_angles,
        markers,
        trimmed_extrema_percent=trimmed_extrema_percent,
    )

    # Update scaling setup file
    scaling_root[0].find("mass").text = str(subject_mass)
    scaling_root[0].find("height").text = str(subject_height)
    scaling_root[0].find("GenericModelMaker").find(
        "model_file"
    ).text = scaled_model_path
    scaling_root[0].find(".//scaling_order").text = " manualScale measurements"
    deactivate_measurements(scaling_root)
    update_scale_values(scaling_root, segment_ratio_dict)
    for mk_f in scaling_root[0].findall(".//marker_file"):
        mk_f.text = "Unassigned"
    scaling_root[0].find("ModelScaler").find(
        "output_model_file"
    ).text = scaled_model_path

    etree.indent(scaling_tree, space="\t", level=0)
    scaling_tree.write(
        scaling_path_temp, pretty_print=True, xml_declaration=True, encoding="utf-8"
    )

    # Run scaling
    opensim.ScaleTool(scaling_path_temp).run()

    # Remove scaling setup
    Path(scaling_path_temp).unlink()


def scale_model(
    # Required experiment parameters
    trc_folder: str,
    scaled_model_path: str,
    # Optional experiment parameters
    subject_height="auto",
    subject_mass=70.0,
    fastest_frames_to_remove_percent=0.2,
    large_hip_knee_angles=0.5,
    trimmed_extrema_percent=0.2,
    close_to_zero_speed=0.5,
):
    """
    Runs OpenSim scaling and inverse kinematics.
    """
    # Determine subject height
    if (
        subject_height is None
        or subject_height == 0
        or (isinstance(subject_height, str) and subject_height.lower() == "auto")
    ):
        print("Automatically computing subject height from TRC data ...")
        trc_data, markers = aggregate_trc_frames(trc_folder)
        subject_height = compute_height(
            trc_data,
            markers,
            fastest_frames_to_remove_percent=fastest_frames_to_remove_percent,
            close_to_zero_speed=close_to_zero_speed,
            large_hip_knee_angles=large_hip_knee_angles,
            trimmed_extrema_percent=trimmed_extrema_percent,
        )
        if not np.isnan(subject_height):
            print(
                f"Subject height automatically calculated for {trc_folder.name}: {round(subject_height, 2)} m\n"
            )
        else:
            print(
                f"Could not compute height from {trc_folder.name}. Using default height of 1.75m."
            )
            print(
                "The person may be static, crouched, or incorrectly detected. Adjust parameters in Config.toml if needed."
            )
            subject_height = 1.75

    # Determine subject mass
    if (
        subject_mass is None
        or not isinstance(subject_mass, (int, float))
        or subject_mass == 0
    ):
        subject_mass = 70.0
        print("No subject mass found. Using default mass of 70.0kg.")

    # Perform scaling
    print(
        f"Scaling model using subject height: {round(subject_height, 2)} m and mass: {round(subject_mass, 1)} kg ...\n"
    )
    print(f"All TRC files in {trc_folder} will be used for scaling.\n")
    return perform_scaling(
        trc_folder,
        scaled_model_path,
        subject_height=subject_height,
        subject_mass=subject_mass,
        fastest_frames_to_remove_percent=fastest_frames_to_remove_percent,
        large_hip_knee_angles=large_hip_knee_angles,
        trimmed_extrema_percent=trimmed_extrema_percent,
        close_to_zero_speed_m=close_to_zero_speed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run OpenSim kinematics from a TRC file and configuration file."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to the folder containing TRC files for scaling.",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to the output scaled OpenSim model file.",
        required=True,
    )
    parser.add_argument(
        "--subject-height",
        type=str,
        default="auto",
        help="Subject height in meters or 'auto' to compute automatically (default: 'auto').",
    )
    parser.add_argument(
        "--subject-mass",
        type=float,
        default=70.0,
        help="Subject mass in kilograms (default: 70.0).",
    )
    parser.add_argument(
        "--fastest_frames_to_remove_percent",
        type=float,
        default=0.2,
        help="Percentage of fastest frames to remove (default: 0.2).",
    )
    parser.add_argument(
        "--large_hip_knee_angles",
        type=float,
        default=0.5,
        help="Threshold for large hip and knee angles in degrees (default: 0.5).",
    )
    parser.add_argument(
        "--trimmed_extrema_percent",
        type=float,
        default=0.2,
        help="Percentage of extreme segment values to trim (default: 0.2).",
    )
    parser.add_argument(
        "--close_to_zero_speed",
        type=float,
        default=0.5,
        help="Threshold for close-to-zero speed in meters/frame (default: 0.5).",
    )
    args = parser.parse_args()

    # Read CLI args.
    trc_folder = Path(args.input).resolve()
    if not trc_folder.is_dir():
        raise NotADirectoryError(f"Input path {trc_folder} is not a valid directory.")

    scaled_model_path = Path(args.output).resolve()
    if scaled_model_path.suffix.lower() != ".osim":
        raise ValueError(
            f"Output model path {scaled_model_path} must have a .osim extension."
        )

    # Run kinematics
    scale_model(
        trc_folder=trc_folder,
        scaled_model_path=scaled_model_path,
        # Optional experiment parameters
        subject_height=args.subject_height,
        subject_mass=args.subject_mass,
        fastest_frames_to_remove_percent=args.fastest_frames_to_remove_percent,
        large_hip_knee_angles=args.large_hip_knee_angles,
        trimmed_extrema_percent=args.trimmed_extrema_percent,
        close_to_zero_speed=args.close_to_zero_speed,
    )
