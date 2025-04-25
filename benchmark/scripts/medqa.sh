#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=4,5,6,7
# python -m medqa.eval_eng enable_calibrate=false
# python -m medqa.eval_eng enable_calibrate=true
# python -m medqa.eval enable_calibrate=false
# python -m medqa.eval enable_calibrate=true
# python -m medqa.eval enable_calibrate=false model_path=Qwen/Qwen2.5-7B-Instruct
# python -m medqa.eval enable_calibrate=true model_path=Qwen/Qwen2.5-7B-Instruct
python -m medqa.eval_eng_gpt enable_calibrate=false debug=true recompute_calibration=true
