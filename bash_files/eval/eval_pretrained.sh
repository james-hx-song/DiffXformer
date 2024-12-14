#!/bin/bash

# Define tasks array with proper bash syntax
TASKS=(
    "arc_challenge"
    "arc_easy"
    "openbookqa"
    "hellaswag"
    "piqa"
    "winogrande"
)

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_type>"
    echo "model_type: Transformer or DiffFormer"
    exit 1
fi

model_type=$1

for task in "${TASKS[@]}"; do
    python eval_model.py --task $task --model $model_type
done
