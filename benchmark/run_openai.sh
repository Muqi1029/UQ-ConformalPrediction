#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=5,6

model_path="gpt-4o-mini"

# dataset_names=("gsm8k" "triviaqa")
dataset_names=("medqa_us")

for dataset_name in "${dataset_names[@]}"; do
    echo "Running: model_path=${model_path}, dataset_name=${dataset_name}"
    python main.py model_path="${model_path}" \
        dataset_name="${dataset_name}" \
        enable_calibrate=true

    echo "Running: model_path=${model_path}, dataset_name=${dataset_name} enable_calibrate=false"
    python main.py model_path="${model_path}" \
        dataset_name="${dataset_name}" \
        enable_calibrate=false
done
