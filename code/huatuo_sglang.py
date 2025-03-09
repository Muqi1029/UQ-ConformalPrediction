import torch
from sglang import Runtime, assistant, function, gen, set_default_backend, system, user
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "/mnt/user/models/HuatuoGPT-o1-7B"


@function
def chat(s, p):
    s += user("What should I do if my head hurts")
    s += assistant(gen(name="first_response", max_tokens=512))
    s += user("Please provide your confidence in the answer only in percent (0–100 %):")
    s += assistant(gen(name="second_response", max_tokens=128))


def predict():
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="bfloat16",
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    prompt = "你是谁？"
    conversation = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        conversation=conversation, add_generation_prompt=True, tokenize=False
    )
    tokenized_inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**tokenized_inputs)
    first_response = tokenizer.decode(
        generated_ids[0, tokenized_inputs["input_ids"].size(1) :]
    )

    conversation.append({"role": "assistant", "content": first_response})
    conversation.append(
        {
            "role": "user",
            "content": "Please provide your confidence in the answer only in percent (0–100 %):",
        }
    )
    chat_prompt = tokenizer.apply_chat_template(
        conversation=conversation, add_generation_prompt=True, tokenize=False
    )
    tokenized_inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    second_generated_ids = model.generate(**tokenized_inputs)
    print(tokenizer.decode(second_generated_ids[0], skip_special_tokens=True))


def test_sglang():
    set_default_backend(Runtime(model_path=model_name_or_path, tp_size=2))
    state = chat.run(
        p="Please provide your confidence in the answer only in percent (0–100 %):"
    )
    # print(state["second_response"])
    print(state.text())


if __name__ == "__main__":
    test_sglang()
    # predict()
