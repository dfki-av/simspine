"""Script to create MMPose-compatible SimSpine dataset from raw HDF5 file."""

import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset


def get_pose_stats(kps):
    """Get statistic information `mean` and `std` of pose data.

    Args:
        kps (ndarray): keypoints in shape [..., K, C] where K and C is
            the keypoint category number and dimension.
    Returns:
        mean (ndarray): [K, C]
    """
    assert kps.ndim > 2
    K, C = kps.shape[-2:]
    kps = kps.reshape(-1, K, C)
    mean = kps.mean(axis=0)
    std = kps.std(axis=0)
    return mean, std


def get_annotations(
    joints_2d, joints_3d, joints_3d_w, scale_factor=1.2, img_size=(1000, 1000)
):
    """Get annotations, including centers, scales, joints_2d and joints_3d.

    Args:
        joints_2d: 2D joint coordinates in shape [N, K, 2], where N is the
            frame number, K is the joint number.
        joints_3d: 3D camera space joint coordinates in shape [N, K, 3], where N is the
            frame number, K is the joint number.
        joints_3d_w: 3D world space joint coordinates in shape [N, K, 3], where N is the
            frame number, K is the joint number.
        scale_factor: Scale factor of bounding box. Default: 1.2.
        img_size: Image size (width, height) used to determine joint visibility.
    Returns:
        centers (ndarray): [N, 2]
        scales (ndarray): [N,]
        joints_2d (ndarray): [N, K, 3]
        joints_3d (ndarray): [N, K, 4]
        joints_3d_w (ndarray): [N, K, 4]
    """
    # calculate joint visibility
    visibility = (
        (joints_2d[:, :, 0] >= 0)
        * (joints_2d[:, :, 0] < img_size[0])
        * (joints_2d[:, :, 1] >= 0)
        * (joints_2d[:, :, 1] < img_size[1])
    )
    visibility = np.array(visibility, dtype=np.float32)[:, :, None]
    joints_2d = np.concatenate([joints_2d, visibility], axis=-1)
    joints_3d = np.concatenate([joints_3d, visibility], axis=-1)
    joints_3d_w = np.concatenate([joints_3d_w, visibility], axis=-1)

    # calculate bounding boxes
    bboxes = np.stack(
        [
            np.min(joints_2d[:, :, 0], axis=1),
            np.min(joints_2d[:, :, 1], axis=1),
            np.max(joints_2d[:, :, 0], axis=1),
            np.max(joints_2d[:, :, 1], axis=1),
        ],
        axis=1,
    )
    centers = np.stack(
        [(bboxes[:, 0] + bboxes[:, 2]) / 2, (bboxes[:, 1] + bboxes[:, 3]) / 2], axis=1
    )
    scales = scale_factor * np.max(bboxes[:, 2:] - bboxes[:, :2], axis=1) / 200

    return centers, scales, joints_2d, joints_3d, joints_3d_w


class RawSimSpineDataset(Dataset):
    SPLITS = {
        "train": ["S1", "S5", "S6", "S7", "S8"],
        "test": ["S9", "S11"],
    }

    MARKERS = [
        "Hip",  # 0, -1
        "Spine_01",  # 1, 0
        "Spine_02",  # 2, 1
        "Spine_03",  # 3, 2
        "Spine_04",  # 4, 3
        "Spine_05",  # 5, 4
        "Neck",  # 6, 5
        "Neck_02",  # 7, 6
        "Neck_03",  # 8, 7
        "Head",  # 9, 8
        "Nose",  # 10, 9
        "LClavicle",  # 11, 5
        "RClavicle",  # 12, 5
        "LLatissimus",  # 13, 4
        "RLatissimus",  # 14, 4
    ]

    KINEMATIC_PARAMS = [
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

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
    ):
        super().__init__()
        assert split in self.SPLITS, f"Unknown split: {split}"

        self.h5_path = h5_path
        self.h5 = None  # will be opened per worker
        self.subjects = self.SPLITS[split]
        self.split = split
        self._build_index()

    def _get_file(self):
        """Open the HDF5 file in read-only mode if not already open (per worker)."""
        if self.h5 is None:
            # SWMR = multiprocessing safe
            self.h5 = h5py.File(self.h5_path, "r", swmr=True)
        return self.h5

    def _build_index(self):
        """Build index structure without opening the file yet."""

        with h5py.File(self.h5_path, "r") as f:
            # Load metadata
            metadata = f["metadata"]
            params = [k.decode() for k in metadata["kinematic_axes"][:]]
            if params != self.KINEMATIC_PARAMS:
                raise ValueError("Kinematic axes do not match expected SimSpine axes.")

            markers = [m.decode() for m in metadata["markers_names"][:]]
            if markers != self.MARKERS:
                raise ValueError("Marker names do not match expected SimSpine markers.")

            root_index = markers.index("Hip")
            parent_ids = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 5, 4, 4]

            # Save metadata info as a class attribute
            self.metadata = {
                "root_name": "Hip",
                "root_index": root_index,
                "joint_names": np.array(markers),
                "param_names": np.array(params),
                "parent_ids": np.array(parent_ids),
            }

            # Collect subject-action pairs
            index = []
            cameras = {}
            for subj in [s for s in f.keys() if s.startswith("S")]:
                if subj not in self.subjects:
                    continue

                # load camera parameters
                cam_names = f[subj]["calibration"]["names"][:]  # [V,]
                size = f[subj]["calibration"]["size"][:]  # [V,2]
                R_all = f[subj]["calibration"]["R"][:]  # [V,3,3]
                t_all = f[subj]["calibration"]["t"][:]  # [V,3]
                K_all = f[subj]["calibration"]["K"][:]  # [V,3,3]
                dist_all = f[subj]["calibration"]["dist"][:]  # [V,D]
                cameras[subj] = []
                for cam in range(len(K_all)):
                    K = K_all[cam]  # [3,3]
                    w, h = size[cam].tolist()
                    cam_param = dict(
                        K=K,
                        R=R_all[cam].reshape(3, 3),
                        T=t_all[cam].reshape(3, 1),
                        c=np.array([[K[0][2]], [K[1][2]]]),
                        f=np.array([[K[0][0]], [K[1][1]]]),
                        dist=dist_all[cam].reshape(-1, 1),
                        w=w,
                        h=h,
                        name=cam_names[cam].decode(),
                    )
                    cameras[subj].append(cam_param)

                for seq in f[subj].keys():
                    if seq == "calibration":
                        continue
                    index.append((subj, seq))

            # Build expanded list of samples: (subject, action, start_idx)
            self.samples = []
            for subj, seq in index:
                for cam_param in cameras[subj]:
                    cam_name = cam_param["name"]
                    prefix = f"{subj}_{seq}.{cam_name}"

                    img_prefix = f"{subj}/{prefix}/{prefix}"
                    self.samples.append((subj, seq, cam_param, img_prefix))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f = self._get_file()  # ensure per-worker file handle
        subj, seq, cam_param, img_prefix = self.samples[idx]
        subj_group = f[subj]
        seq_data = subj_group[seq]

        # load camera parameters
        mtx = cam_param["K"]
        dist = cam_param["dist"]
        R = cam_param["R"]
        rvec, _ = cv2.Rodrigues(R)
        tvec = cam_param["T"].reshape(-1)

        # load angles
        angles = seq_data["kinematics"][:]  # [T, P]

        # load 3D annotations (OpenSim coordinate system)
        joints_3d = seq_data["markers"][:]  # [T , K, 3]
        joints_3d = joints_3d[:, :, [2, 0, 1]]  # YZX → XYZ

        joints_3d_w = joints_3d.reshape(-1, 3)  # [T*K, 3] for projection
        num_frames = joints_3d.shape[0]  # T
        num_joints = joints_3d.shape[1]  # K

        imgname = np.array(
            [f"{img_prefix}_{i:06d}.jpg" for i in range(num_frames)],
        )  # [T,]

        # load 2D annotations
        imgpts, _ = cv2.projectPoints(joints_3d_w, rvec, tvec, mtx, dist)
        joints_2d = imgpts.reshape(-1, num_joints, 2).astype(np.float32)  # [T, K, 2]

        # transform 3D from world to camera
        joints_3d = (R @ (joints_3d_w - tvec).T).T  # [T*K,3]
        joints_3d = joints_3d.reshape(-1, num_joints, 3)

        # save world coordinates as well
        joints_3d_w = joints_3d_w.reshape(-1, num_joints, 3)

        # get centers, scales, and visibility
        centers, scales, joints_2d, joints_3d, joints_3d_w = get_annotations(
            joints_2d, joints_3d, joints_3d_w
        )

        # convert all data to np.float64
        joints_2d = joints_2d.astype(np.float64)
        joints_3d = joints_3d.astype(np.float64)
        joints_3d_w = joints_3d_w.astype(np.float64)
        angles = angles.astype(np.float64)

        return dict(
            imgname=imgname,
            centers=centers,
            scales=scales,
            part=joints_2d,
            S=joints_3d,
            S_world=joints_3d_w,
            angles=angles,
        )

    def close(self):
        try:
            self.h5.close()
        except Exception:
            pass


if __name__ == "__main__":
    import os
    import pickle
    from os.path import join

    from tqdm import tqdm

    data_file = "data/simspine/simspine.h5"
    out_dir = "data/simspine"

    annot_dir = join(out_dir, "annotations")
    os.makedirs(annot_dir, exist_ok=True)

    splits = ["train", "test"]
    for split in splits:
        print(f"Loading {split} split from HDF5...")
        dataset = RawSimSpineDataset(
            h5_path=data_file,
            split=split,
        )

        _imgnames = []
        _centers = []
        _scales = []
        _joints_2d = []
        _joints_3d = []
        _joints_3d_w = []
        _angles = []

        for i in tqdm(range(len(dataset)), desc=f"Processing {split} set"):
            sample = dataset[i]
            _imgnames.append(sample["imgname"])
            _centers.append(sample["centers"])
            _scales.append(sample["scales"])
            _joints_2d.append(sample["part"])
            _joints_3d.append(sample["S"])
            _joints_3d_w.append(sample["S_world"])
            _angles.append(sample["angles"])

        _imgnames = np.concatenate(_imgnames)
        _centers = np.concatenate(_centers)
        _scales = np.concatenate(_scales)
        _joints_2d = np.concatenate(_joints_2d)
        _joints_3d = np.concatenate(_joints_3d)
        _joints_3d_w = np.concatenate(_joints_3d_w)
        _angles = np.concatenate(_angles)

        out_file = join(annot_dir, f"simspine_{split}.npz")
        np.savez(
            out_file,
            metadata=dataset.metadata,
            imgname=_imgnames,
            center=_centers,
            scale=_scales,
            part=_joints_2d,
            S=_joints_3d,
            S_world=_joints_3d_w,
            angles=_angles,
        )
        print(
            f"Create annotation file for trainset: {out_file}. "
            f"{len(_imgnames)} samples in total."
        )

        # Print metadata info
        print("Metadata info:")
        for key, value in dataset.metadata.items():
            print(f"  {key}: {value}")

        if split == "train":
            # get `mean` and `std` of pose data
            _joints_3d = _joints_3d[..., :3]  # remove visibility
            mean_3d, std_3d = get_pose_stats(_joints_3d)

            _joints_2d = _joints_2d[..., :2]  # remove visibility
            mean_2d, std_2d = get_pose_stats(_joints_2d)

            # centered around root
            root_index = dataset.metadata["root_index"]

            root_3d = _joints_3d[..., root_index : root_index + 1, :]
            _joints_3d_rel = _joints_3d - root_3d

            mean_3d_rel, std_3d_rel = get_pose_stats(_joints_3d_rel)
            mean_3d_rel[root_index] = mean_3d[root_index]
            std_3d_rel[root_index] = std_3d[root_index]

            root_2d = _joints_2d[..., root_index : root_index + 1, :]
            _joints_2d_rel = _joints_2d - root_2d

            mean_2d_rel, std_2d_rel = get_pose_stats(_joints_2d_rel)
            mean_2d_rel[root_index] = mean_2d[root_index]
            std_2d_rel[root_index] = std_2d[root_index]

            stats = {
                "joint3d_stats": {"mean": mean_3d, "std": std_3d},
                "joint2d_stats": {"mean": mean_2d, "std": std_2d},
                "joint3d_rel_stats": {"mean": mean_3d_rel, "std": std_3d_rel},
                "joint2d_rel_stats": {"mean": mean_2d_rel, "std": std_2d_rel},
            }
            for name, stat_dict in stats.items():
                out_file = join(annot_dir, f"{name}.pkl")
                with open(out_file, "wb") as f:
                    pickle.dump(stat_dict, f)
                print(f"Create statistic data file: {out_file}")
