from typing import List

from datasets import load_dataset

second_prompt = """
This is the logprobs of the first response:
{avg_logprobs}, lower means the response is more creative or has potential of dangerous, please provide the confidence in the answer only in percent (0–100 %) direcly, then following the rationales:
"""


def load_dataset_util(dataset_name: str):
    if dataset_name == "coqa":
        return load_dataset("coqa", split="train")
    elif dataset_name == "trivia_qa":
        return load_dataset("TimoImhof/TriviaQA-in-SQuAD-format", split="unmodified")
    else:
        raise ValueError(f"Dataset {dataset_name} not found")


def judge(pred, answer: List):
    return answer[0] in pred.lower()
