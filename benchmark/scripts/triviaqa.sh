#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=3 python -m triviaqa.eval enable_calibrate=true
# CUDA_VISIBLE_DEVICES=3 python -m triviaqa.eval enable_calibrate=false
# CUDA_VISIBLE_DEVICES=3 python -m triviaqa.eval model_path=Qwen/Qwen2.5-7B-Instruct enable_calibrate=false
# CUDA_VISIBLE_DEVICES=3 python -m triviaqa.eval model_path=Qwen/Qwen2.5-7B-Instruct enable_calibrate=true

CUDA_VISIBLE_DEVICES=3 python -m triviaqa.eval_gpt enable_calibrate=true model_path=gpt-4o-mini
CUDA_VISIBLE_DEVICES=3 python -m triviaqa.eval_gpt enable_calibrate=false model_path=gpt-4o-mini
