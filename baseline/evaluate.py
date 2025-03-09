import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

with open("baseline/eval.yaml") as fout:
    try:
        config = yaml.safe_load(fout)
    except Exception as e:  # Improved exception handling
        print(f"Error loading configuration: {e}")
        exit(1)  # Exit if config loading fails

device = "cuda" if torch.cuda.is_available() else "cpu"
id2label = {0: "False", 1: "True"}


def run():
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name_or_path"]
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    while True:
        inputs = input("please input a sentence that you want to check: ")
        if inputs.strip().lower() in ["q", "quit"]:  # Simplified input check
            break
        tokenized_inputs = tokenizer(inputs, return_tensors="pt").to(
            device
        )  # Moved to(device) here
        output = model(**tokenized_inputs)
        id = output.logits.argmax(dim=-1).item()
        print(id2label[id])


if __name__ == "__main__":
    run()
