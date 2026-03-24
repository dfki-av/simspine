#!/bin/bash
# Script to create the SIMSPINE dataset from Human3.6M and predicted spine markers
# using the SpinePose models from Khan et al. (CVPR Workshop 2025).

# Processing parameters
SUBJECTS=("S1" "S5" "S6" "S7" "S8" "S9" "S11")  # H3.6M subjects to process
DATA_RATE=50    # Original data frame rate in Hz
CAMERA_RATE=50  # Sampling rate of cameras in Hz

# Preprocessed Human3.6M data directory
# This must exist prior to running this script. See README for instructions.
h36m_dir="data/h36m/processed"
if [ ! -d "$h36m_dir" ]; then
    echo "Error: Human3.6M directory '$h36m_dir' does not exist."
    exit 1
fi

# SIMSPINE dataset directory
simspine_dir="data/simspine"  # Output directory for SIMSPINE dataset
scratch_dir="data/simspine_scratch"  # Scratch directory for intermediate files

# Loop over each subject to create the dataset
for subject in "${SUBJECTS[@]}"; do
    echo "Processing subject: $subject"

    # Subject-specific directories
    # <h36m_dir>/
    #   └── annotations/<Subject>/        (Human3.6M ground truth marker trc files)
    h36m_gt_dir="${h36m_dir}/annotations/${subject}"
    if [ ! -d "$h36m_gt_dir" ]; then
        echo "Error: Human3.6M ground truth for `$subject` not found in '$h36m_gt_dir'. Skipping."
        continue
    fi

    # <simspine_dir>/
    #   ├── kinematics/<Subject>/         (simulated kinematics files)
    #   ├── markers/<Subject>/            (marker trajectory files)
    #   └── models/<Subject>.osim         (scaled OpenSim model for the subject)
    kinematics_dir="${simspine_dir}/kinematics/${subject}"
    markers_dir="${simspine_dir}/markers/${subject}"
    subject_model_file="${simspine_dir}/models/${subject}.osim"

    # <scratch_dir>/
    #   ├── 0_Predicted/<Subject>/   (predicted trc files)
    #   ├── 1_WithKnownGT/<Subject>/ (merged predicted + known GT trc files)
    #   ├── 2_Simulated/<Subject>/   (simulated mot and trc files)
    #   ├── 3_Merged/<Subject>/      (merged simulated spine + known GT trc files)
    scratch00_dir="${scratch_dir}/0_Predicted/${subject}"
    scratch01_dir="${scratch_dir}/1_WithKnownGT/${subject}"
    scratch02_dir="${scratch_dir}/2_Simulated/${subject}"
    scratch03_dir="${scratch_dir}/3_Merged/${subject}"

    if [ ! -d "$scratch00_dir" ]; then
        echo "Error: Predicted spine markers for `$subject` not found in '$scratch00_dir'. Skipping."
        continue
    fi

    # # 1) Merge predicted spine pseudo-markers with known GT body from Human3.6M
    echo "  Merging pseudo-markers with known GT body markers..."
    for file in $h36m_gt_dir/$subject_*.trc; do
        filename=$(basename "$file")
        python src/simspine/data_generation/1_merge_predictions.py \
            "$file" \
            "$scratch00_dir/$filename" \
            "$scratch01_dir/$filename"
        echo "    Merged: $filename"
    done
    echo "    Merged files saved to: $scratch01_dir"

    # # 2) Create scaled OpenSim model for the subject
    echo "  Creating scaled OpenSim model for $subject..."
    python src/simspine/data_generation/2_scale_model.py \
        --input $scratch01_dir/ \
        --output $scratch_dir/Subjects/$subject.osim \
        --subject-height auto \
        --subject-mass 70.0
    echo "    Scaled model saved to: $subject_model_file"

    # 3) Create SIMSPINE dataset by simulating kinematics and markers
    echo "  Creating SIMSPINE dataset for $subject..."
    for file in $scratch01_dir/$subject_*.trc; do
        filename=$(basename "$file")
        action_name="${filename%.*}"
        echo "    Processing action: $action_name"

        3.1) Simulate kinematics using OpenSim
        python src/simspine/data_generation/3_kinematics.py \
            --input $scratch01_dir/$filename \
            --model $scratch_dir/Subjects/$subject.osim \
            --output $kinematics_dir/$action_name
        if [ $? -ne 0 ]; then
            echo "      ✗ Kinematics simulation failed. Aborting $action_name ($subject)."
            continue
        else
            echo "      ✓ Simulated kinematics saved to: $kinematics_dir/$action_name"
        fi

        # 3.2) Simulate marker locations
        python src/simspine/data_generation/4_simulate_markers.py \
            --input $kinematics_dir/$action_name/_ik_model_marker_locations.sto \
            --output $scratch02_dir/$filename \
            --data-rate $DATA_RATE \
            --camera-rate $CAMERA_RATE
        if [ $? -ne 0 ]; then
            echo "      ✗ Marker simulation failed. Aborting $action_name ($subject)."
            continue
        else
            echo "      ✓ Simulated: $scratch02_dir/$filename"
        fi

        # 3.3) Merge simulated markers with pseudo-markers from prediction step (#1 above)
        python src/simspine/data_generation/5_merge_simulation.py \
            --input $scratch01_dir/$filename \
            --simulated $scratch02_dir/$filename \
            --output $scratch03_dir/$filename \
            --data-rate $DATA_RATE \
            --camera-rate $CAMERA_RATE
        if [ $? -ne 0 ]; then
            echo "      ✗ Merging simulated markers failed. Aborting $action_name ($subject)."
            continue
        else
            echo "      ✓ Merged: $scratch03_dir/$filename"
        fi

        # 3.4) Filter the merged marker trajectories to reduce noise
        python src/simspine/data_generation/6_filtering.py \
            --input $scratch03_dir/$filename \
            --output $markers_dir/$filename \
            --order 2 \
            --cutoff 5 \
            --type low \
            --frame-rate $CAMERA_RATE
        if [ $? -ne 0 ]; then
            echo "      ✗ Filtering failed. Aborting $action_name ($subject)."
            continue
        else
            echo "      ✓ Filtered: $markers_dir/$filename"
        fi
    done
done
