from typing import List


def triviaqa_make_prompt(item):
    p = "Question: {question}\nAnswer:"
    return p.format(question=item["question"])


def triviaqa_judge(pred, answer: List[str]):
    return any(ans.lower() in pred.lower() for ans in answer)


def map_triviaqa(example):
    example["answer"] = example["answers"]["text"]
    return example
