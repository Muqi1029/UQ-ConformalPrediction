import json
import logging
import os
import random

import hydra
import numpy as np
from omegaconf import OmegaConf
from sglang import OpenAI, Runtime, assistant, function, gen, set_default_backend, user
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utils import (
    extract_confidence,
    gsm_judge,
    load_dataset_util,
    logger_setup,
    no_calibrate_prompt,
    second_prompt,
    trim_result,
)

OmegaConf.register_new_resolver("get_last_segment", lambda path: path.split("/")[-1])
first = True


def make_prompt(item):
    p = f"Question: {item['question']}\n"
    return p


def compute_rate(avg_logprobs, calibrate_data):
    return np.mean(np.array(calibrate_data) <= avg_logprobs) * 100


def compute_calibration(calibrate_dataset, cfg):
    calibrate_data = []
    batch_size = cfg.batch_size
    num_batches = (len(calibrate_dataset) - 1) // batch_size + 1
    for batch_idx in tqdm(range(num_batches), desc="Computing calibration"):
        start_idx = batch_size * batch_idx
        end_idx = min(start_idx + batch_size, len(calibrate_dataset))
        states = qa.run_batch(
            [
                {"q": make_prompt(calibrate_dataset[i])}
                for i in range(start_idx, end_idx)
            ],
            max_new_tokens=cfg.max_new_tokens,
        )
        for i, s in enumerate(states):
            # only consider the correct answer
            if gsm_judge(s["answer"], calibrate_dataset[start_idx + i]["answer"]):
                avg_logprobs = np.mean(
                    [
                        tok[0]
                        for tok in s.get_meta_info("rationale")["output_token_logprobs"]
                    ]
                )
                calibrate_data.append(avg_logprobs)

    os.makedirs(os.path.dirname(cfg.calibration_file), exist_ok=True)
    with open(cfg.calibration_file, "w", encoding="utf-8") as f:
        json.dump(calibrate_data, f, ensure_ascii=False, indent=4)
    logging.info(
        f"Calibration file saved to {cfg.calibration_file}, there are totally {len(calibrate_data)} samples"
    )
    return calibrate_data


@function
def qa(s, q):
    s += user(expr=q)
    s += assistant(
        gen(name="rationale", return_logprob=True)
        + " So the answer is: "
        + gen(name="answer")
    )


@function
def chat(s, q, calibrate_data, enable_calibrate=True):
    s += user(q)
    s += assistant(
        gen(name="rationale", return_logprob=True)
        + " So the answer is: "
        + gen(name="answer")
    )

    if enable_calibrate:
        avg_logprobs = np.mean(
            [tok[0] for tok in s.get_meta_info("rationale")["output_token_logprobs"]]
        )
        rate = compute_rate(avg_logprobs, calibrate_data)
        s += user(second_prompt.format(rate=rate))
    else:
        s += user(no_calibrate_prompt)
    s += assistant(gen(name="confidence"))


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: OmegaConf):
    logger_setup()
    logging.info(f"Config: {cfg}")
    model_dir = os.path.dirname(cfg.results_file)
    os.makedirs(model_dir, exist_ok=True)

    # load dataset
    dataset = load_dataset_util("gsm8k")
    if cfg.debug:
        logging.info(f"Debug mode, using {cfg.sample_size} samples")
        dataset = dataset.select(range(cfg.sample_size))

    logging.info(f"Loading model from {cfg.model_path}")
    if cfg.model_path == "gpt-4o-mini":
        backend = OpenAI(
            model_name=cfg.model_path,
            api_key=os.environ["API_KEY"],
            base_url=os.environ["BASE_URL"],
        )
    else:
        backend = Runtime(
            model_path=cfg.model_path,
            dp_size=cfg.dp_size,
            tp_size=cfg.tp_size,
            dtype="bfloat16",
        )
    set_default_backend(backend)

    # load calibration
    if not os.path.exists(cfg.calibration_file) or cfg.recompute_calibration:
        logging.info(f"Computing calibration for {cfg.calibration_sample_size} samples")
        calibration_sample_size = min(cfg.calibration_sample_size, len(dataset))
        indices = random.sample(range(len(dataset)), calibration_sample_size)
        calibration_dataset = load_dataset_util("gsm8k", options="calibrate").select(
            indices
        )
        calibrate_data = compute_calibration(calibration_dataset, cfg)
    else:
        logging.info(f"Loading calibration from {cfg.calibration_file}")
        with open(cfg.calibration_file, "r") as f:
            calibrate_data = json.load(f)

    # experiment record
    preds = []  # predicted answers
    answers = []  # ground truth answers
    confidences = []  # extracted calibrated confidence scores
    raw_confidences = []  # raw confidence scores
    valid_count = 0  # number of valid samples

    batch_size = cfg.batch_size
    num_batches = (len(dataset) - 1) // batch_size + 1
    for batch_idx in tqdm(range(num_batches), desc="Running inference"):
        start_idx = batch_size * batch_idx
        end_idx = min(start_idx + batch_size, len(dataset))
        states = chat.run_batch(
            [
                {
                    "q": make_prompt(dataset[i]),
                    "calibrate_data": calibrate_data,
                    "enable_calibrate": cfg.enable_calibrate,
                }
                for i in range(start_idx, end_idx)
            ],
            max_new_tokens=cfg.max_new_tokens,
        )
        for i, s in enumerate(states):

            # save prompt example
            try:
                global first
                if first:
                    if cfg.enable_calibrate:
                        path = os.path.join(model_dir, "prompt_example_calibrate.txt")
                else:
                    path = os.path.join(model_dir, "prompt_example_no_calibrate.txt")
                with open(path, "w+") as f:
                    f.write(s.text())
                first = False

                # extract confidence
                confidence = extract_confidence(s["confidence"])
                if confidence != 0:
                    valid_count += 1

                    # predicted answer(first response)
                    preds.append(trim_result(s["answer"]))

                    # extract raw confidence (second response)
                    raw_confidences.append(s["confidence"])

                    # extract calibrated confidence (float)
                    confidences.append(confidence)

                    # ground truth answer
                    answers.append(dataset[start_idx + i]["answer"])
            except Exception as e:
                logging.error(f"Error in inference: {e}")
                continue

    # compute accuracy and auroc
    acc = [p == a for p, a in zip(preds, answers)]
    auroc = roc_auc_score(np.array(acc), np.array(confidences))

    logging.info(f"Accuracy: {np.mean(acc)}")
    logging.info(f"Enable calibrate: {cfg.enable_calibrate}")
    logging.info(f"AUROC: {auroc}")

    # save results to mdoel dir for better check
    if cfg.enable_calibrate:
        cfg.results_file = cfg.results_file.replace(".json", "_calibrate.json")
    else:
        cfg.results_file = cfg.results_file.replace(".json", "_no_calibrate.json")
    with open(cfg.results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "preds": preds,
                "answers": answers,
                "confidences": confidences,
                "raw_confidences": raw_confidences,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )
    logging.info(f"Results saved to {cfg.results_file}")

    # summary tables
    with open("summary.csv", mode="a+", encoding="utf-8") as f:
        f.write(
            f"gsm8k,{cfg.model_name},{str(cfg.enable_calibrate)},{np.mean(acc)},{auroc},{valid_count / len(dataset)}\n"
        )


if __name__ == "__main__":
    main()
