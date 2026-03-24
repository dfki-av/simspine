import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from .common import (
    ACTIONS,
    DATA_DIR,
    FIGURES_DIR,
    find_marker_files,
    read_marker_data,
)

if __name__ == "__main__":
    FIGURES_DIR = FIGURES_DIR / "distribution"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    SELECTED_MARKERS = [
        "Spine_01",
        "Spine_02",
        "Spine_03",
        "Spine_04",
        "Spine_05",
        "Neck",
        "Neck_02",
        "Neck_03",
        "LClavicle",
        "RClavicle",
        "LShoulder",
        "RShoulder",
        "LHip",
        "RHip",
    ]
    MAX_POINTS = 1000

    LINKS = [
        ("LHip", "Spine_01"),
        ("RHip", "Spine_01"),
        ("Spine_01", "Spine_02"),
        ("Spine_02", "Spine_03"),
        ("Spine_03", "Spine_04"),
        ("Spine_04", "Spine_05"),
        ("Spine_05", "LClavicle"),
        ("Spine_05", "RClavicle"),
        ("LClavicle", "LShoulder"),
        ("RClavicle", "RShoulder"),
        ("Spine_05", "Neck"),
        ("Neck", "Neck_02"),
        ("Neck_02", "Neck_03"),
    ]

    VIEWS = "frontal", "sagittal"
    X_LIM = (-0.4, 0.4)
    Y_LIM = (-0.1, 0.7)

    for action in ACTIONS:
        trc_files = find_marker_files(DATA_DIR, action=action)

        # read all TRC files and create a single numpy array [N*T, K, 3]
        all_coords = []
        all_markers = None
        for trc_file in tqdm(trc_files, desc="Loading TRC files"):
            coords, markers = read_marker_data(trc_file)
            all_coords.append(coords)
            if all_markers is None:
                all_markers = markers
            else:
                assert all_markers == markers, "Markers do not match across TRC files"
        all_coords = np.concatenate(all_coords, axis=0)  # [N*T, K, 3]
        print(f"Loaded {len(trc_files)} TRC files with total shape {all_coords.shape}")

        root_idx = all_markers.index("Hip")
        root_coords = all_coords[:, root_idx, :]

        for view in VIEWS:
            output_dir = os.path.join(FIGURES_DIR, view)
            output_file = os.path.join(output_dir, f"{action}.pdf")
            os.makedirs(output_dir, exist_ok=True)
            if os.path.exists(output_file):
                print(f"Skipping {view} view for {action}, already exists.")
                continue

            # Create figure layout
            g = None
            palette = sns.color_palette("husl", len(SELECTED_MARKERS))

            # Storage for centers
            centers = {}

            # Plot each joint’s density map
            for color, joint_name in zip(palette, SELECTED_MARKERS):
                if joint_name not in all_markers:
                    print(f"⚠️ Warning: {joint_name} not found in markers.")
                    continue

                jidx = all_markers.index(joint_name)
                rel_coords = all_coords[:, jidx, :] - root_coords

                if view == "frontal":
                    df = pd.DataFrame(
                        {
                            "x": rel_coords[:, 2],  # depth
                            "y": rel_coords[:, 0],  # lateral
                            "z": rel_coords[:, 1],  # vertical
                        }
                    )
                else:  # sagittal
                    df = pd.DataFrame(
                        {
                            "x": -rel_coords[:, 0],  # lateral
                            "y": rel_coords[:, 2],  # depth
                            "z": rel_coords[:, 1],  # vertical
                        }
                    )

                # Downsample for speed
                if len(df) > MAX_POINTS:
                    df = df.sample(n=MAX_POINTS, random_state=42).reset_index(drop=True)

                if g is None:
                    g = sns.JointGrid(data=df, x="x", y="z", height=8, ratio=5)

                # Filled 2D KDE
                sns.kdeplot(
                    data=df,
                    x="x",
                    y="z",
                    ax=g.ax_joint,
                    fill=True,
                    cmap=sns.light_palette(color, as_cmap=True),
                    alpha=0.8,
                    levels=25,
                    label=joint_name,
                )

                # Compute and store mean center
                cx, cz = df["x"].mean(), df["z"].mean()
                centers[joint_name] = (cx, cz)
                g.ax_joint.scatter(
                    cx,
                    cz,
                    color=color,
                    s=40,
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=3,
                )
                g.ax_joint.text(
                    cx + 0.015,
                    cz,
                    joint_name.replace("_", ""),
                    color=color,
                    fontsize=9,
                    fontweight="bold",
                    alpha=0.8,
                )

                # Use fixed axis limits for consistency
                g.ax_joint.set_xlim(X_LIM)
                g.ax_joint.set_ylim(Y_LIM)

            # Add origin marker
            g.ax_joint.scatter(
                0, 0, color="black", marker="x", s=100, label="Hip (Origin)"
            )

            # Connect mean centers along approximate spine hierarchy
            for a, b in LINKS:
                if a in centers and b in centers:
                    x1, z1 = centers[a]
                    x2, z2 = centers[b]
                    g.ax_joint.plot(
                        [x1, x2], [z1, z2], color="gray", lw=1, alpha=0.7, zorder=2
                    )

            # Labels and title
            view_name = "Frontal (X-Z)" if view == "frontal" else "Sagittal (Y-Z)"
            if view == "frontal":
                g.set_axis_labels(
                    "lateral (Hip-relative)", "vertical (Hip-relative)", fontsize=14
                )
            else:
                g.set_axis_labels(
                    "depth (Hip-relative)", "vertical (Hip-relative)", fontsize=14
                )

            # Save figure
            plt.tight_layout()
            plt.savefig(
                output_file,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(f"Saved {view_name} view for {action}.")
