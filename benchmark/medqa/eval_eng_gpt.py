import json
import logging
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

import hydra
import numpy as np
from omegaconf import OmegaConf
from openai import OpenAI, OpenAIError
from sklearn.metrics import roc_auc_score
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm
from utils import (
    extract_confidence,
    load_dataset_util,
    logger_setup,
    no_calibrate_prompt,
    second_prompt,
)

client = OpenAI(api_key=os.environ["API_KEY"], base_url=os.environ["BASE_URL"])
model_name = ""
first = True
OmegaConf.register_new_resolver("get_last_segment", lambda path: path.split("/")[-1])


def make_prompt(item):
    p = "Given the question and options, please select the most likely answer from the options.\n"
    p += f"Question: {item['question']}\n"
    for k, v in item["options"].items():
        p += f"Option {k}: {v}\n"
    p += "You MUST provide your answer **strictly** in the format in the end of your response: **'The answer is [X].'** where [X] is ONE uppercase letter from A/B/C/D."
    return p


def compute_rate(avg_logprobs, calibrate_data):
    return np.mean(np.array(calibrate_data) <= avg_logprobs) * 100


def extract_answer(content):
    return re.search(r"[Aa]nswer.*?([ABCD])", content).group(1)


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(OpenAIError),
)
def chat_openai(message, cfg):
    response = client.chat.completions.create(
        model=model_name,
        messages=message,
        max_completion_tokens=cfg.max_new_tokens,
        logprobs=True,
    )
    content = response.choices[0].message.content
    logprobs = response.choices[0].logprobs
    avg_logprobs = np.mean([token.logprob for token in logprobs.content])
    return content, avg_logprobs


def chat_openai_with_calibration(message, calibrate_data, cfg):
    content, avg_logprobs = chat_openai(message, cfg)
    rate = compute_rate(avg_logprobs, calibrate_data)
    message.append({"role": "assistant", "content": content})
    if cfg.enable_calibrate:
        message.append({"role": "user", "content": second_prompt.format(rate=rate)})
        confidence_content, _ = chat_openai(message, cfg)
    else:
        message.append(
            {
                "role": "user",
                "content": no_calibrate_prompt,
            }
        )
        confidence_content, _ = chat_openai(message, cfg)
    return content, confidence_content


def compute_calibration(calibration_dataset, cfg):
    calibrate_data = []

    pbar = tqdm(total=len(calibration_dataset), desc="Computing calibration")
    with ThreadPoolExecutor(max_workers=cfg.num_threads) as executor:
        futures = [
            executor.submit(
                chat_openai,
                message=[{"role": "user", "content": make_prompt(item)}],
                cfg=cfg,
            )
            for item in calibration_dataset
        ]
        for future, item in zip(futures, calibration_dataset):
            pbar.update(1)
            try:
                content, avg_logprobs = future.result()
                # only consider the correct answer
                if extract_answer(content) == item["answer_idx"]:
                    calibrate_data.append(avg_logprobs)
            except Exception as e:
                logging.error(f"Error in calibration: {e}")
                continue

    pbar.close()

    # save calibration data
    os.makedirs(os.path.dirname(cfg.calibration_file), exist_ok=True)
    with open(cfg.calibration_file, "w", encoding="utf-8") as f:
        json.dump(calibrate_data, f, ensure_ascii=False, indent=4)

    logging.info(
        f"Calibration file saved to {cfg.calibration_file}, there are totally {len(calibrate_data)} samples"
    )
    return calibrate_data


@hydra.main(config_path=".", config_name="config_eng", version_base=None)
def main(cfg: OmegaConf):
    logger_setup()
    logging.info("Starting evaluation")

    global model_name
    model_name = cfg.model_path
    model_dir = os.path.dirname(cfg.results_file)
    os.makedirs(model_dir, exist_ok=True)
    logging.info(f"Using config: {cfg}")

    # load dataset
    dataset = load_dataset_util("med_qa", options="US")
    if cfg.debug:
        logging.info(f"Debug mode, using {cfg.sample_size} samples")
        dataset = dataset.select(range(cfg.sample_size))

    # load calibration
    if not os.path.exists(cfg.calibration_file) or cfg.recompute_calibration:
        calibration_sample_size = min(cfg.calibration_sample_size, len(dataset))
        logging.info(f"Computing calibration for {calibration_sample_size} samples")
        indices = random.sample(range(len(dataset)), calibration_sample_size)
        calibration_dataset = dataset.select(indices)
        calibrate_data = compute_calibration(calibration_dataset, cfg)
    else:
        logging.info(f"Loading calibration from {cfg.calibration_file}")
        with open(cfg.calibration_file, "r") as f:
            calibrate_data = json.load(f)

    # experiment record
    preds = []
    answers = []
    confidences = []
    raw_confidences = []
    valid_count = 0

    pbar = tqdm(total=len(dataset), desc="Running inference")

    with ThreadPoolExecutor(max_workers=cfg.num_threads) as executor:
        futures = [
            executor.submit(
                chat_openai_with_calibration,
                message=[{"role": "user", "content": make_prompt(item)}],
                calibrate_data=calibrate_data,
                cfg=cfg,
            )
            for item in dataset
        ]
        for future, item in zip(futures, dataset):
            pbar.update(1)
            try:
                content, confidence_content = future.result()

                global first
                if first:
                    if cfg.enable_calibrate:
                        path = os.path.join(model_dir, "prompt_example_calibrate.txt")
                    else:
                        path = os.path.join(
                            model_dir, "prompt_example_no_calibrate.txt"
                        )
                    with open(path, "w+") as f:
                        f.write(content + "\n" + confidence_content)
                    first = False

                confidence = extract_confidence(confidence_content)
                if confidence != 0:
                    valid_count += 1

                    preds.append(content)
                    raw_confidences.append(confidence_content)
                    confidences.append(confidence)

                    answers.append(item["answer_idx"])
            except Exception as e:
                logging.error(f"Error in inference: {e}")
                continue

    pbar.close()

    # compute accuracy and auroc
    acc = []
    new_confidences = []
    for p, a, c in zip(preds, answers, confidences):
        try:
            acc.append(extract_answer(p) == a)
            new_confidences.append(c)
        except Exception as e:
            logging.error(f"Error in extract_answer: {e}")
            continue

    auroc = roc_auc_score(np.array(acc), np.array(new_confidences))

    logging.info(f"Accuracy: {np.mean(acc)}")
    logging.info(f"Enable calibrate: {cfg.enable_calibrate}")
    logging.info(f"AUROC: {auroc}")

    # save results
    if cfg.enable_calibrate:
        cfg.results_file = cfg.results_file.replace(".json", "_calibrated.json")
    else:
        cfg.results_file = cfg.results_file.replace(".json", "_uncalibrated.json")

    with open(cfg.results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "preds": preds,
                "answers": answers,
                "confidences": confidences,
                "raw_confidences": raw_confidences,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )
    # summary tables
    with open("summary.csv", mode="a+", encoding="utf-8") as f:
        f.write(
            f"medqa_us,{cfg.model_name},{cfg.enable_calibrate},{np.mean(acc)},{auroc},{valid_count / len(dataset)}\n"
        )
    logging.info(f"Results saved to {cfg.results_file}")


if __name__ == "__main__":
    main()
