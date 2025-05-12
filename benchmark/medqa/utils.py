def medqa_make_prompt(item):
    p = "给定问题和选项，请从选项中选择最可能的答案。\n"
    p += f"问题: {item['question']}\n"
    for k, v in item["options"].items():
        p += f"选项 {k}: {v}\n"
    return p


def medqa_us_make_prompt(item):
    p = 'Given the question and options, think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.\n'
    p += f"Question: {item['question']}\n"
    for k, v in item["options"].items():
        p += f"Option {k}: {v}\n"
    return p


def medqa_judge(pred, answer):
    return pred == answer


def map_medqa(example):
    example["answer"] = example["answer_idx"]
    return example
