import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import toml
from matplotlib.animation import FuncAnimation


def read_trc(trc_path):
    """Read TRC file into (frames, time, coords[T,K,3], markers)."""
    with open(trc_path, "r") as f:
        lines = f.readlines()
    header_line, data_start = 3, 5
    header_parts = lines[header_line].strip().split("\t")
    markers = [p for p in header_parts if p not in ["Frame#", "Time", ""]]
    data = np.loadtxt(trc_path, skiprows=data_start)
    frames = data[:, 0].astype(int)
    time = data[:, 1]
    coords = data[:, 2:].reshape(len(frames), len(markers), 3)
    return frames, time, coords, markers


def make_links(markers):
    """Basic Human3.6M-style connectivity."""
    links = [
        ("Hip", "RHip"),
        ("RHip", "RKnee"),
        ("RKnee", "RAnkle"),
        ("Hip", "LHip"),
        ("LHip", "LKnee"),
        ("LKnee", "LAnkle"),
        ("LShoulder", "LElbow"),
        ("LElbow", "LWrist"),
        ("RShoulder", "RElbow"),
        ("RElbow", "RWrist"),
        ("Hip", "Spine_01"),
        ("Spine_01", "Spine_02"),
        ("Spine_02", "Spine_03"),
        ("Spine_03", "Spine_04"),
        ("Spine_04", "Spine_05"),
        ("Spine_05", "Neck"),
        ("Neck", "Neck_02"),
        ("Neck_02", "Neck_03"),
        ("Neck_03", "Head"),
        ("Spine_05", "LClavicle"),
        ("LClavicle", "LShoulder"),
        ("Spine_05", "RClavicle"),
        ("RClavicle", "RShoulder"),
        ("Spine_04", "LLatissimus"),
        ("Spine_04", "RLatissimus"),
        ("LAnkle", "LHeel"),
        ("LAnkle", "LBigToe"),
        ("LBigToe", "LSmallToe"),
        ("RAnkle", "RHeel"),
        ("RAnkle", "RBigToe"),
        ("RBigToe", "RSmallToe"),
        ("Head", "Nose"),
        ("Nose", "LEye"),
        ("Nose", "REye"),
        ("Nose", "LEar"),
        ("Nose", "REar"),
    ]
    return [(a, b) for a, b in links if a in markers and b in markers]


def load_camera(calib_path, cam_name):
    """Load calibration for given camera name from TOML."""
    with open(calib_path, "rb") as f:
        calib = toml.load(f)
    if cam_name not in calib:
        raise KeyError(f"Camera {cam_name} not found in {calib_path}")
    c = calib[cam_name]
    K = np.array(c["matrix"], dtype=np.float32)
    dist = np.array(c["distortions"], dtype=np.float32)
    rvec = np.array(c["rotation"], dtype=np.float32)
    tvec = np.array(c["translation"], dtype=np.float32)
    size = tuple(c["size"])
    return K, dist, rvec, tvec, size


def project_points(P, K, dist, rvec, tvec):
    """Project 3D world points to image points."""
    pts, _ = cv2.projectPoints(P.astype(np.float32), rvec, tvec, K, dist)
    return pts.reshape(-1, 2)


def animate_markers_plot(coords, markers, links, interval=50):
    """Fallback 3D animation with Matplotlib."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    scat = ax.scatter([], [], [], s=25)
    lines = [ax.plot([], [], [], lw=2)[0] for _ in links]

    all_coords = coords.reshape(-1, 3)
    mins, maxs = all_coords.min(0), all_coords.max(0)
    lim = (maxs - mins).max() / 2
    mid = (maxs + mins) / 2
    ax.set_xlim(mid[0] - lim, mid[0] + lim)
    ax.set_ylim(mid[1] - lim, mid[1] + lim)
    ax.set_zlim(mid[2] - lim, mid[2] + lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    def update(i):
        P = coords[i]
        scat._offsets3d = (P[:, 0], P[:, 1], P[:, 2])
        for l, (a, b) in zip(lines, links):
            i1, i2 = markers.index(a), markers.index(b)
            l.set_data([P[i1, 0], P[i2, 0]], [P[i1, 1], P[i2, 1]])
            l.set_3d_properties([P[i1, 2], P[i2, 2]])
        return [scat] + lines

    FuncAnimation(fig, update, frames=len(coords), interval=interval)
    plt.show()


def animate_markers_overlay(video_path, coords, markers, links, K, dist, rvec, tvec):
    """Overlay skeleton directly onto video using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), len(coords))

    print(f"Playing {video_path} ({fps:.1f} FPS) with {total} frames...")

    coords = coords[:, :, [2, 0, 1]]  # Convert from OpenSim to CV coord system

    while True:
        for i in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            P = coords[i]
            uv = project_points(P, K, dist, rvec, tvec)

            # Draw bones
            for a, b in links:
                i1, i2 = markers.index(a), markers.index(b)
                p1, p2 = tuple(uv[i1].astype(int)), tuple(uv[i2].astype(int))
                cv2.line(frame, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw joints
            for x, y in uv.astype(int):
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

            cv2.imshow("Skeleton Overlay", frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animate TRC in 3D or overlay on a video."
    )
    parser.add_argument(
        "trc",
        type=str,
        help="Path to TRC file",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video",
    )
    parser.add_argument(
        "--calib",
        type=str,
        default=None,
        help="Path to camera calibration TOML",
    )
    args = parser.parse_args()

    frames, time, coords, markers = read_trc(args.trc)
    links = make_links(markers)

    if args.video and args.calib:
        cam_name = Path(args.video).stem
        K, dist, rvec, tvec, size = load_camera(args.calib, cam_name)
        animate_markers_overlay(
            args.video,
            coords,
            markers,
            links,
            K,
            dist,
            rvec,
            tvec,
        )
    else:
        animate_markers_plot(coords, markers, links)
