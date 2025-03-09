from functools import partial
from pprint import pprint

import yaml
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def compute_metrics(
    outputs,
):
    logits, labels = outputs
    predictions = logits.argmax(axis=-1)
    # Compute metrics
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    precision = precision_score(y_true=labels, y_pred=predictions, average="binary")
    recall = recall_score(y_true=labels, y_pred=predictions, average="binary")
    f1 = f1_score(y_true=labels, y_pred=predictions, average="binary")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    return model, tokenizer


def load_custom_dataset(dataset_name):
    if dataset_name == "arithmetic":
        return load_dataset(
            "json",
            data_files={
                "train": "data/arithmetic/train.jsonl",
                "test": "data/arithmetic/test.jsonl",
            },
        )
    elif dataset_name == "imdb":
        pass
    else:
        raise NotImplementedError(f"{dataset_name} has not been implemented.")


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["input"], truncation=True)


def train():
    model, tokenizer = load_model_and_tokenizer()
    ds = load_custom_dataset(config["dataset_name"]).shuffle()
    func = partial(preprocess_function, tokenizer=tokenizer)
    encoded_dataset = ds.map(func, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        learning_rate=2e-5,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epoch"],
        weight_decay=0.01,
        report_to="wandb",
    )

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 开始训练
    trainer.train()


with open("baseline/train.yaml") as fout:
    config = yaml.safe_load(fout)
    pprint(config)

if __name__ == "__main__":
    train()
