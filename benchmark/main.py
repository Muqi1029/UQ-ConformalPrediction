import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import hydra
import numpy as np
from gsm8k.utils import gsm_judge, gsm_make_prompt
from medqa.utils import medqa_judge, medqa_us_make_prompt
from omegaconf import OmegaConf
from openai import OpenAI, OpenAIError
from sglang import Runtime, assistant, function, gen, select, set_default_backend, user
from sklearn.metrics import roc_auc_score
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm
from triviaqa.utils import triviaqa_judge, triviaqa_make_prompt
from utils import (
    compute_calibration,
    compute_rate,
    extract_choice,
    extract_confidence,
    load_dataset_util,
    logger_setup,
    seed_everything,
)

OmegaConf.register_new_resolver("get_last_segment", lambda path: path.split("/")[-1])
first = True
client = None

no_calibrate_prompt = "Please directly give a confidence score for your answer between 0 and 100, and explain your confidence score. For example, Confidence: 90, Explanation: The answer is correct because..."
second_prompt = """
This is the quantile of the logprobs of the first response:
{rate}, lower means the response is more creative or has potential of dangerous, please provide the confidence in the answer only in percent (0–100 %) direcly, then following the rationales:
"""


@function
def qa(s, q, stage: str, calibrate_data=None, enable_calibrate=False):
    s += user(q)
    s += assistant(
        gen(name="rationale", return_logprob=True)
        + "\nSo the answer is "
        + gen(name="answer", temperature=0.0)
    )

    if stage == "inference":
        if enable_calibrate:
            avg_logprobs = np.mean(
                [
                    tok[0]
                    for tok in s.get_meta_info("rationale")["output_token_logprobs"]
                ]
            )
            rate = compute_rate(avg_logprobs, calibrate_data)
            s += user(second_prompt.format(rate=rate))
        else:
            s += user(no_calibrate_prompt)
        s += assistant(gen(name="confidence", temperature=0.0))


@function
def choice(s, q, stage: str, calibrate_data=None, enable_calibrate=False):
    s += user(q)
    s += assistant(
        gen(name="rationale", return_logprob=True)
        + "\nSo the answer is "
        + select(name="answer", choices=["A", "B", "C", "D"])
    )
    if stage == "inference":
        if enable_calibrate:
            avg_logprobs = np.mean(
                [
                    tok[0]
                    for tok in s.get_meta_info("rationale")["output_token_logprobs"]
                ]
            )
            rate = compute_rate(avg_logprobs, calibrate_data)
            s += user(second_prompt.format(rate=rate))
        else:
            s += user(no_calibrate_prompt)
        s += assistant(gen(name="confidence", temperature=0.0))


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(OpenAIError),
)
def chat_openai(message: List[Dict[str, str]], cfg):
    response = client.chat.completions.create(
        model=cfg.model_path,
        messages=message,
        max_completion_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        logprobs=True,
    )
    content = response.choices[0].message.content
    logprobs = response.choices[0].logprobs
    avg_logprobs = np.mean([token.logprob for token in logprobs.content])
    return content, avg_logprobs


def two_turn_prompt(message, cfg):
    content, avg_logprobs = chat_openai(message, cfg)
    message.append({"role": "assistant", "content": content})
    message.append(
        {
            "role": "user",
            "content": "\nSo the answer is ",
        }
    )
    answer_content, _ = chat_openai(message, cfg)
    return answer_content, avg_logprobs


def chat_openai_with_calibration(
    message: List[Dict[str, str]],
    stage: str,
    cfg: OmegaConf,
    calibrate_data=None,
):
    answer_content, avg_logprobs = two_turn_prompt(message, cfg)

    if stage == "inference":
        rate = compute_rate(avg_logprobs, calibrate_data)

        message.append({"role": "assistant", "content": answer_content})
        if cfg.enable_calibrate:
            message.append({"role": "user", "content": second_prompt.format(rate=rate)})
        else:
            message.append(
                {
                    "role": "user",
                    "content": no_calibrate_prompt,
                }
            )
        confidence_content, _ = chat_openai(message, cfg)
        return answer_content, confidence_content
    elif stage == "calibration":
        return answer_content, avg_logprobs
    else:
        raise ValueError(f"Invalid stage: {stage}")


def compute_calibration_openai(
    calibration_dataset, make_prompt_function, judge_function, cfg
):
    calibrate_data = []

    with tqdm(total=len(calibration_dataset), desc="Computing calibration") as pbar:
        with ThreadPoolExecutor(max_workers=cfg.num_threads) as executor:
            futures = [
                executor.submit(
                    chat_openai_with_calibration,
                    message=[{"role": "user", "content": make_prompt_function(item)}],
                    stage="calibration",
                    cfg=cfg,
                )
                for item in calibration_dataset
            ]
            try:
                for future, item in zip(futures, calibration_dataset):
                    answer_content, avg_logprobs = future.result()
                    if judge_function(answer_content, item["answer"]):
                        calibrate_data.append(avg_logprobs)
            except Exception as e:
                logging.error(f"Error in calibration: {e}")
            finally:
                pbar.update(1)

    with open(
        os.path.join(cfg.save_dir, "calibration.json"), mode="w", encoding="utf-8"
    ) as f:
        json.dump(calibrate_data, f, ensure_ascii=False, indent=4)

    logging.info(
        f"Calibration file saved to {os.path.join(cfg.save_dir, 'calibration.json')}, there are totally {len(calibrate_data)} samples"
    )
    return calibrate_data


def run_openai(dataset, make_prompt_function, judge_function, cfg):
    if (
        not os.path.exists(os.path.join(cfg.save_dir, "calibration.json"))
        or cfg.recompute_calibration
    ):
        logging.info(f"Computing calibration for {cfg.calibration_sample_size} samples")
        calibration_sample_size = min(cfg.calibration_sample_size, len(dataset))
        indices = random.sample(range(len(dataset)), calibration_sample_size)
        calibrate_dataset = dataset.select(indices)
        calibrate_data = compute_calibration_openai(
            calibrate_dataset,
            make_prompt_function,
            judge_function,
            cfg,
        )
    else:
        logging.info(
            f"Loading calibration from {os.path.join(cfg.save_dir, 'calibration.json')}"
        )
        with open(os.path.join(cfg.save_dir, "calibration.json"), "r") as f:
            calibrate_data = json.load(f)

    valid_count = 0
    save_jsons = []
    with tqdm(total=len(dataset), desc="Running inference") as pbar:
        with ThreadPoolExecutor(max_workers=cfg.num_threads) as executor:
            futures = [
                executor.submit(
                    chat_openai_with_calibration,
                    message=[{"role": "user", "content": make_prompt_function(item)}],
                    stage="inference",
                    cfg=cfg,
                    calibrate_data=calibrate_data,
                )
                for item in dataset
            ]

            for future, item in zip(futures, dataset):
                try:
                    answer_content, confidence_content = future.result()

                    global first
                    if first:
                        if cfg.enable_calibrate:
                            path = os.path.join(
                                cfg.save_dir, "prompt_example_calibrate.txt"
                            )
                        else:
                            path = os.path.join(
                                cfg.save_dir, "prompt_example_no_calibrate.txt"
                            )
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(answer_content + "\n" + confidence_content)
                        first = False

                    confidence = extract_confidence(confidence_content)
                    if confidence != 0:
                        valid_count += 1
                        if cfg.dataset_name == "medqa_us":
                            pred = extract_choice(answer_content)
                        else:
                            pred = answer_content
                        save_jsons.append(
                            {
                                "completed_text": answer_content,
                                "confidence": confidence,
                                "answer": item["answer"],
                                "is_correct": judge_function(pred, item["answer"]),
                            }
                        )
                except Exception as e:
                    logging.error(f"Error in inference: {e}")
                    continue
                finally:
                    pbar.update(1)

    return valid_count, save_jsons


def run_sgl(dataset, sgl_function, make_prompt, judge_function, cfg):
    if (
        not os.path.exists(os.path.join(cfg.save_dir, "calibration.json"))
        or cfg.recompute_calibration
    ):
        calibration_sample_size = min(cfg.calibration_sample_size, len(dataset))
        logging.info(f"Computing calibration for {calibration_sample_size} samples")
        indices = random.sample(range(len(dataset)), calibration_sample_size)
        calibrate_dataset = dataset.select(indices)
        calibrate_data = compute_calibration(
            sgl_function,
            make_prompt,
            judge_function,
            calibrate_dataset,
            cfg,
        )
    else:
        logging.info(
            f"Loading calibration from {os.path.join(cfg.save_dir, 'calibration.json')}"
        )
        with open(os.path.join(cfg.save_dir, "calibration.json"), "r") as f:
            calibrate_data = json.load(f)

    # experiment record
    save_jsons = []
    valid_count = 0

    num_batches = (len(dataset) - 1) // cfg.batch_size + 1
    for batch_idx in tqdm(range(num_batches), desc="Running inference"):
        start_idx = cfg.batch_size * batch_idx
        end_idx = min(start_idx + cfg.batch_size, len(dataset))
        states = sgl_function.run_batch(
            [
                {
                    "q": make_prompt(dataset[i]),
                    "stage": "inference",
                    "calibrate_data": calibrate_data,
                    "enable_calibrate": cfg.enable_calibrate,
                }
                for i in range(start_idx, end_idx)
            ],
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )
        for i, s in enumerate(states):
            # save prompt example
            try:
                global first
                if first:
                    if cfg.enable_calibrate:
                        path = os.path.join(
                            cfg.save_dir, "prompt_example_calibrate.txt"
                        )
                    else:
                        path = os.path.join(
                            cfg.save_dir, "prompt_example_no_calibrate.txt"
                        )
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(s.text())
                    first = False

                # extract confidence
                confidence = extract_confidence(s["confidence"])
                if confidence != 0:
                    valid_count += 1

                    save_jsons.append(
                        {
                            "completed_text": s.text(),
                            "confidence": confidence,
                            "answer": dataset[start_idx + i]["answer"],
                            "is_correct": judge_function(
                                s["answer"], dataset[start_idx + i]["answer"]
                            ),
                        }
                    )
            except Exception as e:
                logging.error(f"Error in inference: {e}")
                continue

    return valid_count, save_jsons


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: OmegaConf):
    logger_setup()
    logging.info(f"Config: {cfg}")
    seed_everything(cfg.seed)

    # results_dir/model_name/dataset_name/
    save_dir = os.path.join(
        cfg.results_dir,
        cfg.model_name,
        cfg.dataset_name,
    )
    cfg.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    # load dataset
    dataset = load_dataset_util(cfg.dataset_name)
    if cfg.dataset_name == "gsm8k":
        sgl_function = qa
        judge_function = gsm_judge
        make_prompt = gsm_make_prompt
    elif cfg.dataset_name == "medqa_us":
        sgl_function = choice
        judge_function = medqa_judge
        make_prompt = medqa_us_make_prompt
    elif cfg.dataset_name == "triviaqa":
        sgl_function = qa
        judge_function = triviaqa_judge
        make_prompt = triviaqa_make_prompt
    else:
        raise ValueError(f"Dataset {cfg.dataset_name} not supported")

    if cfg.debug:
        logging.info(f"Debug mode, using {cfg.sample_size} samples")
        dataset = dataset.select(range(cfg.sample_size))

    logging.info(f"Loading model from {cfg.model_path}")
    if cfg.model_path == "gpt-4o-mini":
        global client
        client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
        )
        valid_count, save_jsons = run_openai(dataset, make_prompt, judge_function, cfg)
    else:
        backend = Runtime(
            model_path=cfg.model_path,
            dp_size=cfg.dp_size,
            tp_size=cfg.tp_size,
            dtype="bfloat16",
        )
        set_default_backend(backend)
        valid_count, save_jsons = run_sgl(
            dataset, sgl_function, make_prompt, judge_function, cfg
        )

    # compute accuracy and auroc
    acc = [item["is_correct"] for item in save_jsons]
    auroc = roc_auc_score(
        np.array(acc), np.array([item["confidence"] for item in save_jsons])
    )

    logging.info(f"Accuracy: {np.mean(acc)}")
    logging.info(f"Enable calibrate: {cfg.enable_calibrate}")
    logging.info(f"AUROC: {auroc}")

    # save results to mdoel dir for better check
    with open(
        os.path.join(save_dir, f"results_calibrate_{cfg.enable_calibrate}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "acc": np.mean(acc),
                "auroc": auroc,
                "valid_count": valid_count,
                "data": save_jsons,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )
    logging.info(
        f"Results saved to {os.path.join(save_dir, f'results_calibrate_{cfg.enable_calibrate}.json')}"
    )

    # summary tables
    with open(
        os.path.join(cfg.results_dir, "summary.csv"), mode="a+", encoding="utf-8"
    ) as f:
        f.write(
            f"{cfg.dataset_name},{cfg.model_name},{str(cfg.enable_calibrate)},{np.mean(acc)},{auroc},{valid_count / len(dataset)}\n"
        )


if __name__ == "__main__":
    main()
