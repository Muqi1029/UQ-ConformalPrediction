from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name_or_path = "/online1/ycsc_wangbenyou/liqi1/models/HuatuoGPT2-7B"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

prompt_accuracy = """
告诉我回复的正确性：

句子： 
准确度

"""


def predict():
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype="bfloat16", trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    prompt = "你是谁？"
    conversation = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        conversation=conversation, add_generation_prompt=True, tokenize=False
    )
    print(chat_prompt)
    tokenized_inputs = tokenizer(chat_prompt, return_tensors="pt")
    for k, v in tokenized_inputs.items():
        tokenized_inputs[k] = v.to(device)
    outputs = model.generate(**tokenized_inputs)
    response = tokenizer.decode(outputs[0, tokenized_inputs["input_ids"].size(dim=1):], skip_special_tokens=True)
    conversation.append({"role": "assistant", "content": response})

    conversation.append({"role": "user", "content": prompt_accuracy})
    


if __name__ == "__main__":
    predict()
