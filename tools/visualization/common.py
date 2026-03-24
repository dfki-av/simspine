import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set global plotting style
sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
    }
)

# Constants
DATA_DIR = Path("data/simspine/markers")
FIGURES_DIR = Path("data/simspine_scratch/figures")
SUBJECTS = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]
ACTIONS = [
    "Directions",
    "Discussion",
    "Eating",
    "Greeting",
    "Phoning",
    "Posing",
    "Purchases",
    "Sitting",
    "Smoking",
    "TakingPhoto",
    "Waiting",
    "Walking",
]


def find_marker_files(base_dir, action):
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(base_dir)
        for f in files
        if f.endswith(".trc") and action in f
    ]


def read_marker_data(trc_path):
    with open(trc_path, "r") as f:
        lines = f.readlines()
    markers = [
        p for p in lines[3].strip().split("\t") if p not in ["Frame#", "Time", ""]
    ]
    data = np.loadtxt(trc_path, skiprows=5)
    coords = data[:, 2:].reshape(len(data), len(markers), 3)
    return coords, markers
