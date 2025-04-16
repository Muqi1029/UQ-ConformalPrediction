import json

import hydra
from sglang import Runtime, assistant, function, gen, set_default_backend, user
from tqdm import tqdm
from utils import judge, load_dataset_util, second_prompt


@function
def chat(s, q):
    s += user(q)
    s += assistant(gen(name="answer", return_logprob=True))

    logprobs = sum(
        [tok[0] for tok in s.get_meta_info("answer")["output_token_logprobs"]]
    )

    s += user(second_prompt.format(avg_logprobs=logprobs))
    s += assistant(gen(name="confidence", max_tokens=32))


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg):
    # load dataset
    dataset = load_dataset_util("trivia_qa")

    set_default_backend(Runtime(model_path=cfg.model_path))

    # batch_run
    preds = []
    answers = []
    confidences = []

    batch_size = cfg.batch_size
    num_batches = (len(dataset) - 1) // batch_size + 1
    for i in tqdm(range(num_batches)):
        start_idx = batch_size * i
        end_idx = min(start_idx + batch_size, len(dataset))
        states = chat.run_batch(
            [{"q": dataset[i]["question"]} for i in range(start_idx, end_idx)],
            max_new_tokens=cfg.max_new_tokens,
        )
        preds.extend([s["answer"] for s in states])
        confidences.extend([s["confidence"] for s in states])
        answers.extend(
            [dataset[i]["answers"]["text"] for i in range(start_idx, end_idx)]
        )

    # compute acc
    acc = sum([judge(p, a) for p, a in zip(preds, answers)]) / len(preds)
    print(f"Accuracy: {acc}")

    # compute uncertainty
    # confidences = [float(u) for u in confidences]
    # print(f"Average uncertainty: {sum(confidences) / len(confidences)}")
    with open("triviaqa/triviaqa_results.json", "w") as f:
        json.dump(
            {
                "preds": preds,
                "answers": answers,
                "confidences": confidences,
                "acc": acc,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
