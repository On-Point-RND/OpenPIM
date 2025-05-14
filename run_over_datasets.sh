#!/bin/bash

# Common arguments that stay the same across all runs
COMMON_ARGS="--lr 0.0001 --batch_size 2048 --n_back 50 --PIM_backbone cond_leak_linear --config_path ./local_configs/exp_egor.yaml"

# Function to run experiments for a specific dataset path and names
run_experiments() {
    local path="$1"
    shift
    local names=("$@")
    
    for name in "${names[@]}"; do
        echo "Running experiment with dataset: $name"
        echo "Dataset path: $path"
        python main.py $COMMON_ARGS \
            --dataset_path "$path" \
            --dataset_name "$name"
        echo "----------------------------------------"
    done
}

# SYNTH 1 Experiments
SYNTH1_PATH="/home/dev/public-datasets/e.shvetsov/PIM/FOR_COOPERATION/"
SYNTH1_NAMES=(
    "16TR_C25Nc16CD_CL_E20Ne1CD_20250117_1L"
    "16TR_C25Nc16CD_CL_E20Ne1CD_20250117_16L"
    "1TR_C20Nc1CD_E20Ne1CD_20250117_0.5m"
    "1TR_C20Nc1CD_E20Ne1CD_20250117_5m"
)
run_experiments "$SYNTH1_PATH" "${SYNTH1_NAMES[@]}"

# SYNTH 2 Experiments
SYNTH2_PATH="/home/dev/public-datasets/e.shvetsov/PIM/data_cooperation_21.04.25/artificial_data_cooperation/"
SYNTH2_NAMES=(
    "16TR_C22Nc8CD_OTX_CL_E20Ne1CD_20250421_16L"
    "16TR_C22Nc8CD_OTX_CL_E20Ne1CD_20250421_1L"
)
run_experiments "$SYNTH2_PATH" "${SYNTH2_NAMES[@]}"

# SYNTH 3 Experiments
SYNTH3_PATH="/home/dev/public-datasets/e.shvetsov/PIM/synt_01/16TR/"
SYNTH3_NAMES=(
    "16TR_C22Nc4CD_CL_E20Ne1CD_20250331_16L"
    "16TR_C22Nc4CD_CL_E20Ne1CD_20250331_1L"
    "16TR_C22Nc8CD_CL_E20Ne1CD_20250331_16L"
    "16TR_C22Nc8CD_CL_E20Ne1CD_20250331_1L"
)
run_experiments "$SYNTH3_PATH" "${SYNTH3_NAMES[@]}"

# # REAL 1 Experiments
# REAL1_PATH="/home/dev/public-datasets/e.shvetsov/PIM/data_cooperation_21.04.25/real_data_cooperation/1TR"
# REAL1_NAMES=("data_A" "data_B")
# run_experiments "$REAL1_PATH" "${REAL1_NAMES[@]}"

# # REAL 2 Experiments
# REAL2_PATH="/home/dev/public-datasets/e.shvetsov/PIM/REAL/Real_data/16TR"
# REAL2_NAMES=("data_16TR_0" "data_16TR_1" "data_16TR_2" "data_16TR_3")
# run_experiments "$REAL2_PATH" "${REAL2_NAMES[@]}"