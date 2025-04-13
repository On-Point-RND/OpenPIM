#!/bin/bash

# Check if the user provided a model name as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

# Assign the first argument to the variable `model_name`
model_name=$1

# Loop over n_back values from 10 to 128 with a step size of 16
for n_back in $(seq 10 16 128); do
    # Run the Python script with the specified arguments
    python main.py \
        --lr 0.0001 \
        --batch_size 64 \
        --n_back $n_back \
        --exp_name "pareto_SYNTH-${model_name}-${n_back}_16TR" \
        --PIM_backbone "$model_name"
done
