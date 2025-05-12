import json
import logging
import os
import re
from typing import Callable

import numpy as np
from datasets import load_dataset
from gsm8k.utils import map_gsm8k
from medqa.utils import map_medqa
from tqdm import tqdm
from triviaqa.utils import map_triviaqa


def logger_setup():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_dataset_util(dataset_name: str):
    if dataset_name == "coqa":
        return load_dataset("coqa", split="train")
    elif dataset_name == "trivia_qa":
        ds = load_dataset("TimoImhof/TriviaQA-in-SQuAD-format", split="unmodified")
        ds = ds.map(map_triviaqa, batched=False)
    elif dataset_name == "medqa_china":
        ds = load_dataset(
            "json",
            data_files={"test": "data/medqa/questions/Mainland/4_options/test.jsonl"},
            split="test",
        )
        ds = ds.map(map_medqa, batched=False)
    elif dataset_name == "medqa_us":
        ds = load_dataset(
            "json",
            data_files={
                "test": "data/medqa/questions/US/4_options/phrases_no_exclude_test.jsonl"
            },
            split="test",
        )
        ds = ds.map(map_medqa, batched=False)
    elif dataset_name == "gsm8k":
        ds = load_dataset(path="openai/gsm8k", name="main", split="test")
        ds = ds.map(map_gsm8k, batched=False)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    return ds


def extract_confidence(confidence: str):
    match = re.search(r"\d+\.\d+|\d+", confidence)

    if match:
        first_number = match.group()
        return float(first_number)
    else:
        return 0.0


def compute_calibration(
    sgl_function: Callable,
    make_prompt: Callable,
    judge_function: Callable,
    calibrate_dataset,
    cfg,
):
    calibrate_data = []
    batch_size = cfg.batch_size
    num_batches = (len(calibrate_dataset) - 1) // batch_size + 1
    for batch_idx in tqdm(range(num_batches), desc="Computing calibration"):
        start_idx = batch_size * batch_idx
        end_idx = min(start_idx + batch_size, len(calibrate_dataset))
        states = sgl_function.run_batch(
            [
                {
                    "q": make_prompt(calibrate_dataset[i]),
                    "stage": "calibration",
                }
                for i in range(start_idx, end_idx)
            ],
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
        )
        for i, s in enumerate(states):
            if judge_function(s["answer"], calibrate_dataset[start_idx + i]["answer"]):
                avg_logprobs = np.mean(
                    [
                        tok[0]
                        for tok in s.get_meta_info("rationale")["output_token_logprobs"]
                    ]
                )
                calibrate_data.append(avg_logprobs)

    with open(
        os.path.join(cfg.save_dir, "calibration.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(calibrate_data, f, ensure_ascii=False, indent=4)

    logging.info(
        f"Calibration file saved to {os.path.join(cfg.save_dir, 'calibration.json')}, there are totally {len(calibrate_data)} samples"
    )
    return calibrate_data


def compute_rate(avg_logprobs, calibrate_data):
    return np.mean(np.array(calibrate_data) <= avg_logprobs) * 100
