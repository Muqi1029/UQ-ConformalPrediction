import re
from typing import Union


def gsm_make_prompt(example):
    return f"Question: {example['question']}\nAnswer:"


def trim_result(output: str) -> Union[str, int]:
    """tranfer an output string to an exact number"""

    # replace numbers like `x,xxx` with `xxxx`
    output = re.sub(r"(\d),(\d)", r"\1\2", output)

    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)

    return numbers[0] if numbers else None


def gsm_judge(pred, answer):
    pred = trim_result(pred)
    if pred is None:
        return False
    return pred == answer


def map_gsm8k(example):
    example["answer"] = example["answer"].split("####")[1].strip()
    # some numbers are in the `x,xxx` format, and we want to remove the comma
    example["answer"] = float(
        re.sub(pattern=r"(\d),(\d)", repl=r"\1\2", string=example["answer"])
    )
    return example
