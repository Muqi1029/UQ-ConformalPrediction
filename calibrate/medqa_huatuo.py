import json
import re
from functools import partial

import torch.nn.functional as F
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from vllm import LLM, SamplingParams


class MedQADataLoader(DataLoader):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass


with open("test_config.yaml") as f:
    config = yaml.safe_load(f)


def load_model_and_tokenizer(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype="bfloat16",
        trust_remote_code=True,
    )
    return model, tokenizer


def create_prompt(example):
    question = example["question"].strip("（　）") + "\n"
    options = "Options:\n"
    for k, v in example["options"].items():
        options += f"{k}. {v}\n"
    return question + options


def extract_answer(text):
    pattern = r"正确答案是选项 \[?([a-dA-D])\]?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    pattern = r"s*([A-Da-d])"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[a-dA-D]\b(?!.*\b[a-dA-D]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def map_function(example, tokenizer):
    chat_conv = [
        {"role": "user", "content": create_prompt(example)},
    ]
    chat_inputs = tokenizer.apply_chat_template(
        chat_conv, add_generation_prompt=True, tokenize=False
    )
    return {"chat_inputs": chat_inputs}


def compute_log_prob(model, input_ids, assistant_length: int):
    labels = input_ids[0, -assistant_length:]
    print(labels.shape)
    outputs = model(input_ids)
    logits = outputs.logits[0, -assistant_length - 1 : -1]
    probs = F.log_softmax(logits, dim=-1)
    avg_logprob = probs[range(assistant_length), labels].mean().item()
    return avg_logprob


def compute_logprob_calibration():
    model, tokenizer = load_model_and_tokenizer(config["model_name_or_path"])
    with open("output.json", "r") as fout:
        true_data = json.load(fout)
    with open("calibration.txt", "a") as fout:
        for item in tqdm(true_data):
            input_words = item["chat_inputs"] + item["model_output"]
            input_ids = tokenizer.encode(input_words, return_tensors="pt").to(
                model.device
            )
            assistant_length = len(tokenizer.encode(item["model_output"]))
            fout.write(f"{compute_log_prob(model, input_ids, assistant_length)}\n")


def main():
    model, tokenizer = load_model_and_tokenizer(config["model_name_or_path"])
    ds = load_dataset("json", data_files=config["data_path"], split="train")

    # this is for test
    if "max_examples" in config:
        ds = ds.shuffle().select(range(config["max_examples"]))

    partial_map_function = partial(map_function, tokenizer=tokenizer)
    ds = ds.map(partial_map_function)

    # compute num_batches
    length = len(ds)
    num_batches = (length - 1) // config["batch_size"] + 1

    true_outputs = []

    # run inference
    for batch in tqdm(range(num_batches)):
        start = batch * config["batch_size"]
        end = (batch + 1) * config["batch_size"]
        inputs = ds[start:end]["chat_inputs"]
        tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(
            model.device
        )
        generated_ids = model.generate(
            **tokenized_inputs, max_new_tokens=config["max_new_tokens"]
        )

        input_length = tokenized_inputs["input_ids"].size(dim=1)
        responses = tokenizer.batch_decode(
            generated_ids[:, input_length:], skip_special_tokens=True
        )
        for i, res in enumerate(responses):
            if extract_answer(res) == ds[start + i]["answer_idx"]:
                example = ds[i]
                example["model_output"] = res
                true_outputs.append(example)
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(true_outputs, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # main()
    compute_logprob_calibration()
