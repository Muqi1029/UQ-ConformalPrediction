import typer
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = typer.Typer()


@app.command()
def classify():
    with open("baseline/eval.yaml") as fout:
        try:
            config = yaml.safe_load(fout)
        except Exception as e:
            typer.echo(f"Error loading configuration: {e}")
            raise typer.Exit(code=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    id2label = {0: "False", 1: "True"}

    # 加载模型和分词器
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name_or_path"]
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])

    while True:
        inputs = typer.prompt("Please input a sentence (or type 'q' to quit)")
        if inputs.strip().lower() in ["q", "quit"]:
            typer.echo("Exiting...")
            break

        tokenized_inputs = tokenizer(inputs, return_tensors="pt").to(device)
        output = model(**tokenized_inputs)
        id = output.logits.argmax(dim=-1).item()
        typer.echo(f"Prediction: {id2label[id]}")


if __name__ == "__main__":
    app()
