import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "results/checkpoint-130"
device = "cuda" if torch.cuda.is_available() else "cpu"
max_new_tokens = 100
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).half().to(device)


def chat_with_model(user_input, history=[]):
    if not user_input.strip():
        return "Please provide a valid input.", history, ""

    # Tokenize input
    tokenized_inputs = tokenizer(user_input, return_tensors="pt", truncation=True)
    tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
        logits = outputs.logits
        prob = torch.sigmoid(logits).squeeze(0)  # Squeeze to remove batch dimension
    # Log probabilities
    print(f"Predicted probabilities: {prob}")

    # Append to history
    response = f"Probability: {prob[1].item():.2%}"  # Example response format
    history.append([user_input, response])

    return history, history, ""


# Define the Gradio interface
if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("## Uncertainty Quantification of **Bert-Math**")
        chatbot = gr.Chatbot(type="messages")
        with gr.Row():
            user_input = gr.Textbox(
                show_label=False,
                placeholder="Input math content to check...",
            )
        history_state = gr.State([])

        # Submit user input
        user_input.submit(
            chat_with_model,
            inputs=[user_input, history_state],
            outputs=[chatbot, history_state, user_input],
        )

    # Launch Gradio interface
    demo.launch()
