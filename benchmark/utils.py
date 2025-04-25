import logging
import re
from typing import List, Union

from datasets import load_dataset


def trim_result(output: str) -> Union[str, int]:
    """tranfer an output string to an exact number"""

    # replace numbers like `x,xxx` with `xxxx`
    output = re.sub(r"(\d),(\d)", r"\1\2", output)

    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)

    return numbers[0] if numbers else None


def map_function(example):
    example["answer"] = example["answer"].split("####")[1].strip()
    # some numbers are in the `x,xxx` format, and we want to remove the comma
    example["answer"] = float(
        re.sub(pattern=r"(\d),(\d)", repl=r"\1\2", string=example["answer"])
    )
    return example


def logger_setup():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


no_calibrate_prompt = "Please directly give a confidence score for your answer between 0 and 100, and explain your confidence score. For example, Confidence: 90, Explanation: The answer is correct because..."
second_prompt = """
This is the quantile of the logprobs of the first response:
{rate}, lower means the response is more creative or has potential of dangerous, please provide the confidence in the answer only in percent (0–100 %) direcly, then following the rationales:
"""


def load_dataset_util(dataset_name: str, options=None):
    if dataset_name == "coqa":
        return load_dataset("coqa", split="train")
    elif dataset_name == "trivia_qa":
        return load_dataset("TimoImhof/TriviaQA-in-SQuAD-format", split="unmodified")
    elif dataset_name == "med_qa":
        if options is None:
            return load_dataset(
                "json",
                data_files={
                    "test": "data/medqa/questions/Mainland/4_options/test.jsonl"
                },
                split="test",
            )
        elif options == "US":
            return load_dataset(
                "json",
                data_files={
                    "test": "data/medqa/questions/US/4_options/phrases_no_exclude_test.jsonl"
                },
                split="test",
            )
        else:
            raise ValueError(f"Dataset {dataset_name} with option {options} not found")
    elif dataset_name == "gsm8k":
        if options is None:
            dataset = load_dataset(path="openai/gsm8k", name="main", split="test")
            dataset = dataset.map(map_function, batched=False)
            return dataset
        elif options == "calibrate":
            dataset = load_dataset(path="openai/gsm8k", name="main", split="train")
            dataset = dataset.map(map_function, batched=False)
            return dataset
        else:
            raise ValueError(f"Dataset {dataset_name} with option {options} not found")
    else:
        raise ValueError(f"Dataset {dataset_name} not found")


def judge(pred, answer: List):
    return answer[0] in pred.lower()


def gsm_judge(pred: str, answer: float):
    pred = trim_result(pred)
    if pred is None:
        return False
    return pred == answer


def extract_confidence(confidence: str):
    match = re.search(r"\d+\.\d+|\d+", confidence)

    if match:
        first_number = match.group()
        return float(first_number)
    else:
        return 0.0
