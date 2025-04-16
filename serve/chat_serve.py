import logging
from pprint import pprint
from threading import Thread

import gradio as gr
import hydra
import markdown
import numpy as np
import pymdownx.emoji
import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from utils import (
    clear_history,
    compute_log_prob,
    disable_btn,
    downvote_last_response,
    enable_btn,
    get_ip,
    regenerate,
    upvote_last_response,
)

model = None
tokenizer = None
calibrate_set = None


def build_about():
    with open("../README.md") as f:
        # Create Markdown instance with GitHub-flavored Markdown extensions
        md = markdown.Markdown(
            extensions=[
                "markdown.extensions.fenced_code",
                "markdown.extensions.tables",
                "markdown.extensions.codehilite",
                "markdown.extensions.sane_lists",
                "markdown.extensions.nl2br",
                "markdown.extensions.meta",
                "pymdownx.emoji",
                "pymdownx.tasklist",
            ],
            extension_configs={
                "pymdownx.emoji": {
                    "emoji_index": pymdownx.emoji.twemoji,
                    "emoji_generator": pymdownx.emoji.to_svg,
                    "options": {
                        "attributes": {
                            "align": "absmiddle",
                            "height": "20px",
                            "width": "20px",
                            "style": "display: inline-block; vertical-align: middle;",
                        },
                    },
                },
                "pymdownx.tasklist": {
                    "custom_checkbox": True,
                    "clickable_checkbox": True,
                },
            },
        )
        # Convert markdown to HTML
        html_content = md.convert(f.read())
        # Add CSS to match GitHub's styling
        styled_html = f"""
        <style>
            .emojione {{
                display: inline-block !important;
                vertical-align: middle !important;
                margin: 0 !important;
            }}
            p > svg {{
                display: inline-block !important;
                vertical-align: middle !important;
                margin: 0 !important;
            }}
            .task-list-item {{
                list-style-type: none !important;
                margin-left: -2.5em !important;
                padding-left: 2.5em !important;
            }}
            .task-list-item input[type="checkbox"] {{
                position: relative;
                margin: 0 0.8em 0.25em -1.6em;
                vertical-align: middle;
                appearance: none;
                -webkit-appearance: none;
                border: 1px solid #d0d7de;
                border-radius: 0.25em;
                width: 1em;
                height: 1em;
                cursor: pointer;
                background-color: #fff;
            }}
            .task-list-item input[type="checkbox"]:checked {{
                background-color: #2da44e;
                border-color: #2da44e;
            }}
            .task-list-item input[type="checkbox"]:checked::after {{
                content: "";
                position: absolute;
                left: 50%;
                top: 45%;
                width: 0.25em;
                height: 0.5em;
                border: solid white;
                border-width: 0 2px 2px 0;
                transform: rotate(45deg);
            }}
            .task-list-control {{
                display: inline !important;
            }}
            ul.contains-task-list {{
                padding-left: 2em !important;
            }}
        </style>
        {html_content}
        """
        return gr.HTML(styled_html, elem_id="about_markdown")


def build_single_model_ui():
    # store history conversation and model components
    state = gr.State([])
    tmp_input = gr.State("")

    # title
    gr.Markdown(
        "## HuatuoGPT2-7B: Uncertainty Quantification", elem_id="notice_markdown"
    )

    # chatbox
    with gr.Group(elem_id="share-region-named"):
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            label="Scroll down and start chatting",
            type="messages",
            height=650,
            show_copy_button=True,
            latex_delimiters=[
                {"left": "$", "right": "$", "display": False},
                {"left": "$$", "right": "$$", "display": True},
                {"left": r"\(", "right": r"\)", "display": False},
                {"left": r"\[", "right": r"\]", "display": True},
            ],
        )

    # textbox + send_btn
    with gr.Row():
        text_input = gr.Textbox(
            show_label=False,
            placeholder="👉 Enter your prompt and press ENTER",
            elem_id="input_box",
        )
        send_btn = gr.Button(value="Send", variant="primary", scale=0)

    # upvote & downvote & regenerate & clear history
    with gr.Row() as button_row:
        upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
        downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
        regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
        clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

    # parameters
    with gr.Accordion("Parameters", open=False) as parameter_row:
        temperature = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.7,
            step=0.1,
            interactive=True,
            label="Temperature",
        )
        top_p = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.1,
            interactive=True,
            label="Top P",
        )
        max_output_tokens = gr.Slider(
            minimum=16,
            maximum=2048,
            value=1024,
            step=64,
            interactive=True,
            label="Max output tokens",
        )

    # Register listeners
    btn_list = [upvote_btn, downvote_btn, regenerate_btn, clear_btn]

    upvote_btn.click(
        upvote_last_response,
        inputs=[state],
        outputs=[text_input, upvote_btn, downvote_btn],
    )

    downvote_btn.click(
        downvote_last_response,
        inputs=[state],
        outputs=[text_input, upvote_btn, downvote_btn],
    )

    regenerate_btn.click(
        regenerate,
        inputs=[state],
        outputs=[state, chatbot, tmp_input, text_input] + btn_list,
    ).then(
        fn=bot_response,
        inputs=[
            state,
            tmp_input,
            temperature,
            top_p,
            max_output_tokens,
        ],
        outputs=[state, chatbot] + btn_list,
    )

    clear_btn.click(
        clear_history, inputs=None, outputs=[state, chatbot, text_input] + btn_list
    )

    # submit
    text_input.submit(
        fn=lambda x: ("", x),
        inputs=[text_input],
        outputs=[text_input, tmp_input],
    ).then(
        fn=bot_response,
        inputs=[
            state,
            tmp_input,
            temperature,
            top_p,
            max_output_tokens,
        ],
        outputs=[state, chatbot] + btn_list,
    )

    send_btn.click(
        fn=lambda x: ("", x),
        inputs=[text_input],
        outputs=[text_input, tmp_input],
    ).then(
        fn=bot_response,
        inputs=[
            state,
            tmp_input,
            temperature,
            top_p,
            max_output_tokens,
        ],
        outputs=[state, chatbot] + btn_list,
    )


def bot_response(
    history_state: gr.State,
    text_input: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    request: gr.Request,
):
    pprint(f"history_state: {history_state}")
    if not text_input:
        return (history_state, history_state) + (enable_btn,) * 4

    ip = get_ip(request)
    logging.info(f"bot_response. ip: {ip}")

    # Tokenize input
    convs = []
    if history_state:
        for message in history_state:
            if message["role"] == "assistant":
                content = message["content"].split("Confidence:")[0]
            else:
                content = message["content"]
            convs.append({"role": message["role"], "content": content})
    # convs and history_state are not the same, convs removes the confidence
    convs.append({"role": "user", "content": text_input})
    history_state.append({"role": "user", "content": text_input})

    # tokenize input
    chat_inputs = tokenizer.apply_chat_template(
        conversation=convs, add_generation_prompt=True, tokenize=False
    )
    tokenized_inputs = tokenizer(chat_inputs, return_tensors="pt", truncation=True).to(
        model.device
    )

    # streamer
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, decode_kwargs={"skip_special_tokens": True}
    )
    generation_kwargs = dict(
        **tokenized_inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    history_state.append({"role": "assistant", "content": ""})

    for new_text in streamer:
        history_state[-1]["content"] += new_text
        yield history_state, history_state, disable_btn, disable_btn, disable_btn, disable_btn

    thread.join()

    # compute confidence
    total_input_ids = tokenizer.apply_chat_template(
        conversation=history_state, add_generation_prompt=True, tokenize=True
    )
    total_input_ids = torch.tensor(total_input_ids, device=model.device).unsqueeze(0)
    assistant_length = tokenizer(history_state[-1]["content"], return_tensors="pt")[
        "input_ids"
    ].size(-1)
    logprob = compute_log_prob(
        model, input_ids=total_input_ids, assistant_length=assistant_length
    )
    position = np.searchsorted(calibrate_set, logprob)
    percentile = position / len(calibrate_set)

    history_state[-1]["content"] += f"\nConfidence: {percentile:.2%}"
    yield history_state, history_state, enable_btn, enable_btn, enable_btn, enable_btn


def build_demo():
    text_size = gr.themes.sizes.text_lg
    with gr.Blocks(
        title="HuatuoGPT2-7B - Uncertainty Quantification",
        theme=gr.themes.Default(text_size=text_size),
    ) as demo:
        with gr.Tabs() as inner_tabs:
            with gr.Tab("💬 Direct Chat", id=2) as direct_tab:
                direct_tab.select(None, None, None)
                build_single_model_ui()
            with gr.Tab("ℹ️ README", id=4):
                build_about()
    return demo


@hydra.main(config_path=".", config_name="serve", version_base="1.2")
def main(config: DictConfig):
    global model, tokenizer, calibrate_set
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
    )
    calibrate_set = np.sort(np.loadtxt(config.calibrate_set_path))

    demo = build_demo()
    demo.launch(
        share=config.share,
        server_name=config.server_name,
        server_port=config.server_port,
    )


if __name__ == "__main__":
    main()
