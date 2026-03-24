import argparse
import logging
from pathlib import Path

import numpy as np
import opensim
import pandas as pd
from lxml import etree

logger = logging.getLogger(__name__)

OPENSIM_SETUP_DIR = Path("assets/OpenSim_Setup")
IK_SETUP_PATH = OPENSIM_SETUP_DIR / "IK_Setup.xml"


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


def perform_IK(
    trc_file: str,
    scaled_model: str,
    output_dir: str,
):
    """
    Perform inverse kinematics based on a TRC file and a scaled OpenSim model:
    - Model markers follow the triangulated markers while respecting the model kinematic constraints
    - Joint angles are computed

    INPUTS:
    - trc_file (Path): The path to the TRC file.
    - scaled_model (Path): The path to the scaled OpenSim model file.
    - output_dir (Path): The directory where the kinematics files are saved.

    OUTPUTS:
    - A joint angle data file (.mot).
    """
    trc_file = Path(trc_file).resolve()
    scaled_model = Path(scaled_model).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Perform IK
    logger.info(f"Running inverse kinematics for {trc_file.name}...")

    try:
        # Retrieve data
        ik_path = IK_SETUP_PATH
        ik_path_temp = str(output_dir / (trc_file.stem + "_ik_setup.xml"))
        output_motion_file = Path(output_dir, trc_file.stem + ".mot").resolve()
        if not trc_file.exists():
            raise FileNotFoundError(f"TRC file does not exist: {trc_file}")
        _, _, time_col, _, _ = read_trc(trc_file)
        start_time, end_time = time_col.iloc[0], time_col.iloc[-1]

        # Update IK setup file
        ik_tree = etree.parse(ik_path)
        ik_root = ik_tree.getroot()
        ik_root.find(".//model_file").text = str(scaled_model)
        ik_root.find(".//time_range").text = f"{start_time} {end_time}"
        ik_root.find(".//output_motion_file").text = str(output_motion_file)
        ik_root.find(".//marker_file").text = str(trc_file.resolve())
        ik_tree.write(ik_path_temp)

        # Run IK
        opensim.InverseKinematicsTool(str(ik_path_temp)).run()

        # Remove IK setup
        Path(ik_path_temp).unlink()

    except Exception as e:
        logger.error(f"Error during IK for {trc_file}: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run OpenSim kinematics from a TRC file and configuration file."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to the input TRC file.",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Path to the output directory for kinematics files.",
        required=True,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        help="Path to the scaled OpenSim model file.",
        required=True,
    )
    args = parser.parse_args()

    perform_IK(
        trc_file=args.input,
        scaled_model=args.model,
        output_dir=args.output,
    )
