#!/bin/bash

# Check if model name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 sep_nlin_mult_infl_fix_pwr"
    exit 1
fi

MODEL_NAME=$1

# Define PIM types
PIM_TYPES=("total" "cond" "ext")

# Define datasets - extracting from data directory structure
DATASETS=(
    "16TR_C22Nc4CD_CL_E20Ne1CD_20250331_16L"
    # "16TR_C22Nc4CD_CL_E20Ne1CD_20250331_1L"
    "16TR_C22Nc8CD_CL_E20Ne1CD_20250331_16L"
    # "16TR_C22Nc8CD_CL_E20Ne1CD_20250331_1L"
    "16TR_C25Nc16CD_CL_E20Ne1CD_20250117_16L"
    # "16TR_C25Nc16CD_CL_E20Ne1CD_20250117_1L"
)

# Function to run experiment
run_experiment() {
    local model=$1
    local dataset=$2
    local pim_type=$3
    local data_path=$4
    
    echo "========================================================"
    echo "Running experiment for:"
    echo "Model: $model"
    echo "Dataset: $dataset"
    echo "PIM Type: $pim_type"
    echo "Data Path: $data_path"
    
    # Run the experiment with pyrallis config
    python model_run.py --model $model --dataset_name $dataset --PIM_type $pim_type
    
    echo "Completed experiment for $dataset with PIM type $pim_type"
    echo "========================================================"
}

echo "Starting experiments with model: $MODEL_NAME"

# Run for synthetic datasets
for dataset in "${DATASETS[@]}"; do
    for pim_type in "${PIM_TYPES[@]}"; do
        run_experiment "$MODEL_NAME" "$dataset" "$pim_type" "../data/"
    done
done

echo "All experiments completed for model: $MODEL_NAME" 
