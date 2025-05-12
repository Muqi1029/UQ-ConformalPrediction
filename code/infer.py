import os
from pprint import pprint

import torch
from openai import OpenAI
from sglang import Runtime, assistant, function, gen, set_default_backend, system, user
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# model_name_or_path = "deepseek-chat"
# model_name_or_path = "gpt-4o-mini"
# model_name_or_path = "gpt-4.1-2025-04-14"

first_prompt = "What should I do if my head hurts"
second_prompt = """
This is the logprobs of the first response:
{avg_logprobs}, lower means the response is more creative or has potential of dangerous, please provide the confidence in the answer only in percent (0–100 %) direcly, then following the rationales:
"""

api_chat_models = [
    "gpt-4.1-2025-04-14",
    "gpt-4o-mini",
    "deepseek-reasoner",
    "deepseek-chat",
]


@function
def sglang_chat(s):
    s += user(first_prompt)
    s += assistant(
        gen(
            name="first_response",
            max_tokens=32,
            return_logprob=True,
            top_logprobs_num=2,
            return_text_in_logprobs=True,
            temperature=1,
        )
    )
    logprobs = s.get_meta_info("first_response")["output_token_logprobs"]
    pprint(s.get_meta_info("first_response")["output_top_logprobs"])

    sum_logprobs = sum([logprob[0] for logprob in logprobs]) / len(logprobs)

    s += user(second_prompt.format(avg_logprobs=sum_logprobs))
    s += assistant(gen(name="second_response", max_tokens=2048))


def transformers_chat():
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="bfloat16",
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    # tokenize the first prompt
    conversation = [{"role": "user", "content": first_prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        conversation=conversation, add_generation_prompt=True, tokenize=False
    )
    tokenized_inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    # generate the first response
    generated_ids = model.generate(**tokenized_inputs)
    first_response = tokenizer.decode(
        generated_ids[0, tokenized_inputs["input_ids"].size(1) :]
    )

    # second turn: tokenize the second prompt
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

    # generate the second response
    second_generated_ids = model.generate(**tokenized_inputs)
    second_response = tokenizer.decode(
        second_generated_ids[0, tokenized_inputs["input_ids"].size(1) :]
    )
    print(second_response)


def test_sglang():
    set_default_backend(Runtime(model_path=model_name_or_path))
    state = sglang_chat.run(stream=True)
    for text in state.text_iter():
        print(text, end="", flush=True)
    pprint(state.get_meta_info("first_response"))


def test_api_chat():
    api_key = (
        os.environ["API_KEY"]
        if "gpt" in model_name_or_path
        else os.environ["DEEPSEEK_API_KEY"]
    )
    base_url = (
        os.environ["BASE_URL"]
        if "gpt" in model_name_or_path
        else os.environ["DEEPSEEK_BASE_URL"]
    )
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    response = client.chat.completions.create(
        model=model_name_or_path,
        messages=[{"role": "user", "content": first_prompt}],
        logprobs=True,
        max_completion_tokens=1024,
    )

    first_response = response.choices[0].message.content
    logprobs = [token.logprob for token in response.choices[0].logprobs.content]
    avg_logprobs = sum(logprobs) / len(logprobs)

    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": first_prompt},
            {
                "role": "assistant",
                "content": first_response,
            },
            {
                "role": "user",
                "content": second_prompt.format(avg_logprobs=avg_logprobs),
            },
        ],
        model=model_name_or_path,
        stream=True,
        max_completion_tokens=128,
    )
    for chunk in response:
        print(chunk.choices[0].delta.content, end="", flush=True)


if __name__ == "__main__":
    if model_name_or_path in api_chat_models:
        test_api_chat()
    else:
        test_sglang()
    # predict()
