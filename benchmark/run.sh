#!/usr/bin/env bash

# python main.py model_path="Qwen/Qwen2.5-7B-Instruct" dataset_name="gsm8k" tp_size=2 recompute_calibration=true enable_calibrate=true batch_size=330 max_new_tokens=32768 calibrate_way="all"
model_paths=("Qwen/Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "gpt-4o-mini")

dataset_names=("gsm8k" "medqa_us" "triviaqa")

for model_path in "${model_paths[@]}"; do
    for dataset_name in "${dataset_names[@]}"; do
        echo "Running: model_path=${model_path}, dataset_name=${dataset_name} enable_calibrate=true"
        python main.py model_path="${model_path}" \
            dataset_name="${dataset_name}" \
            tp_size=2 \
            enable_calibrate=true \
            batch_size=330 \
            max_new_tokens=32768

        echo "Running: model_path=${model_path}, dataset_name=${dataset_name} enable_calibrate=false"
        python main.py model_path="${model_path}" \
            dataset_name="${dataset_name}" \
            tp_size=2 \
            enable_calibrate=false \
            batch_size=330 \
            max_new_tokens=32768
    done
done
