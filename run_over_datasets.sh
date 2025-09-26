#!/bin/bash

# Common arguments that stay the same across all runs
COMMON_ARGS="--lr 0.01 --batch_size 2048 --devices 0 --config_path ./local_configs/exp_egor.yaml"

# List of PIM types to iterate over
PIM_TYPES=("cond")    #("total" "cond" "leak" "ext")

# Function to run experiments for a specific dataset path, names, and PIM type
run_experiments() {
    local pim_type="$1"
    local path="$2"
    shift 2
    local names=("$@")
    
    for name in "${names[@]}"; do
        echo "Running experiment with dataset: $name, PIM Type: $pim_type"
        echo "Dataset path: $path"
        
        # Build full experiment name
        local full_dataset_name="${name}"
        local exp_name="cosine_has_filter_2048_5L_rl_0.01_${pim_type}"

        python main.py $COMMON_ARGS \
            --PIM_backbone "learn_nonlin" \
            --dataset_path "$path" \
            --dataset_name "$full_dataset_name" \
            --exp_name $exp_name \
            --PIM_type $pim_type
        echo "----------------------------------------"
    done
}

# SYNTH 1 Experiments
SYNTH1_PATH="/home/dev/public-datasets/e.shvetsov/PIM/FOR_COOPERATION/"
SYNTH1_NAMES=(
    #"16TR_C25Nc16CD_CL_E20Ne1CD_20250117_1L"
    "16TR_C25Nc16CD_CL_E20Ne1CD_20250117_16L"
    #"1TR_C20Nc1CD_E20Ne1CD_20250117_0.5m"
    #"1TR_C20Nc1CD_E20Ne1CD_20250117_5m"
)

# SYNTH 2 Experiments
SYNTH2_PATH="/home/dev/public-datasets/e.shvetsov/PIM/data_cooperation_21.04.25/artificial_data_cooperation/"
SYNTH2_NAMES=(
    "16TR_C22Nc8CD_OTX_CL_E20Ne1CD_20250421_16L"
    #"16TR_C22Nc8CD_OTX_CL_E20Ne1CD_20250421_1L"
)

# SYNTH 3 Experiments
SYNTH3_PATH="/home/dev/public-datasets/e.shvetsov/PIM/synt_01/16TR/"
SYNTH3_NAMES=(
    # "16TR_C22Nc4CD_CL_E20Ne1CD_20250331_16L"
    #"16TR_C22Nc4CD_CL_E20Ne1CD_20250331_1L"
    "16TR_C22Nc8CD_CL_E20Ne1CD_20250331_16L"
    #"16TR_C22Nc8CD_CL_E20Ne1CD_20250331_1L"
)

# REAL 1 Experiments
REAL1_PATH="/home/dev/public-datasets/e.shvetsov/PIM/data_cooperation_21.04.25/real_data_cooperation/1TR"
REAL1_NAMES=("data_A")  #("data_A" "data_B")

# REAL 2 Experiments
REAL2_PATH="/home/dev/public-datasets/e.shvetsov/PIM/REAL/Real_data/16TR"
REAL2_NAMES=("data_16TR_0" "data_16TR_1" "data_16TR_2" "data_16TR_3")

# Outer loop over PIM types
for PIM_TYPE in "${PIM_TYPES[@]}"; do
    echo "Starting experiments for PIM Type: $PIM_TYPE"

   # run_experiments "$PIM_TYPE" "$SYNTH1_PATH" "${SYNTH1_NAMES[@]}"
    # run_experiments "$PIM_TYPE" "$SYNTH2_PATH" "${SYNTH2_NAMES[@]}"
   run_experiments "$PIM_TYPE" "$SYNTH3_PATH" "${SYNTH3_NAMES[@]}"
    
    # Uncomment below if you want to enable real data runs
    # run_experiments "$PIM_TYPE" "$REAL1_PATH" "${REAL1_NAMES[@]}"
    # run_experiments "$PIM_TYPE" "$REAL2_PATH" "${REAL2_NAMES[@]}"
    
    echo "Finished experiments for PIM Type: $PIM_TYPE"
    echo "========================================"
done