import json
import logging
import os
import random

import hydra
import numpy as np
from sglang import Runtime, assistant, function, gen, set_default_backend, user
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utils import extract_confidence, judge, load_dataset_util, second_prompt


def logger_setup():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info(f"current path: {os.getcwd()}")


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
                {"q": calibrate_dataset[i]["question"]}
                for i in range(start_idx, end_idx)
            ],
            max_new_tokens=cfg.max_new_tokens,
        )
        for i, s in enumerate(states):
            # only consider the correct answer
            if judge(s["answer"], calibrate_dataset[start_idx + i]["answers"]["text"]):
                avg_logprobs = np.mean(
                    [
                        tok[0]
                        for tok in s.get_meta_info("answer")["output_token_logprobs"]
                    ]
                )
                calibrate_data.append(avg_logprobs)

    with open(cfg.calibration_file, "w", encoding="utf-8") as f:
        json.dump(calibrate_data, f, ensure_ascii=False, indent=4)
    logging.info(
        f"Calibration file saved to {cfg.calibration_file}, there are totally {len(calibrate_data)} samples"
    )
    return calibrate_data


@function
def qa(s, q):
    s += user(q)
    s += assistant(gen(name="answer", return_logprob=True))
    return s


@function
def chat(s, q, calibrate_data):
    s += user(q)
    s += assistant(gen(name="answer", return_logprob=True))

    avg_logprobs = np.mean(
        [tok[0] for tok in s.get_meta_info("answer")["output_token_logprobs"]]
    )
    rate = compute_rate(avg_logprobs, calibrate_data)

    s += user(second_prompt.format(rate=rate))
    s += assistant(gen(name="confidence", max_tokens=1024))


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg):
    logger_setup()

    # load dataset
    dataset = load_dataset_util("trivia_qa")
    if cfg.debug:
        logging.info(f"Debug mode, using {cfg.sample_size} samples")
        dataset = dataset.select(range(cfg.sample_size))

    logging.info(f"Loading model from {cfg.model_path}")
    set_default_backend(Runtime(model_path=cfg.model_path))

    # load calibration
    if not os.path.exists(cfg.calibration_file) or cfg.recompute_calibration:
        logging.info(f"Computing calibration for {cfg.calibration_sample_size} samples")
        calibration_sample_size = min(cfg.calibration_sample_size, len(dataset))
        indices = random.sample(range(len(dataset)), calibration_sample_size)
        calibration_dataset = dataset.select(indices)
        calibrate_data = compute_calibration(calibration_dataset, cfg)
    else:
        logging.info(f"Loading calibration from {cfg.calibration_file}")
        with open(cfg.calibration_file, "r") as f:
            calibrate_data = json.load(f)

    # batch_run
    preds = []
    answers = []
    confidences = []
    valid_count = 0

    batch_size = cfg.batch_size
    num_batches = (len(dataset) - 1) // batch_size + 1
    for batch_idx in tqdm(range(num_batches), desc="Running inference"):
        start_idx = batch_size * batch_idx
        end_idx = min(start_idx + batch_size, len(dataset))
        states = chat.run_batch(
            [
                {
                    "q": dataset[i]["question"],
                    "calibrate_data": calibrate_data,
                }
                for i in range(start_idx, end_idx)
            ],
            max_new_tokens=cfg.max_new_tokens,
        )
        for i, s in enumerate(states):
            confidence = extract_confidence(s["confidence"])
            valid_count += 1
            if confidence != 0:
                preds.append(s["answer"])
                confidences.append(confidence)
                answers.append(dataset[start_idx + i]["answers"]["text"])

    # compute acc
    acc = [judge(p, a) for p, a in zip(preds, answers)]
    auroc = roc_auc_score(np.array(acc), np.array(confidences))

    logging.info(f"Accuracy: {np.mean(acc)}")
    logging.info(f"AUROC: {auroc}")

    # save results
    with open(cfg.results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "preds": preds,
                "answers": answers,
                "confidences": confidences,
                "acc": acc,
                "auroc": auroc,
                "valid_rate": valid_count / len(dataset),
            },
            f,
            indent=4,
            ensure_ascii=False,
        )
    logging.info(f"Results saved to {cfg.results_file}")


if __name__ == "__main__":
    main()
