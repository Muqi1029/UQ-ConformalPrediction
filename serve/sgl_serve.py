import json
import logging
from pprint import pprint
from typing import Dict, List

import gradio as gr
import hydra
import numpy as np
from omegaconf import DictConfig
from sglang import Runtime, assistant, function, gen, set_default_backend, user

from serve.utils import (
    clear_history,
    disable_btn,
    downvote_last_response,
    enable_btn,
    get_ip,
    regenerate,
    upvote_last_response,
)

second_prompt = """
This is the quantile of the logprobs of the last response:
{rate}, lower means the response is more creative or has potential of dangerous, please don't talk about the logprobs,just provide the confidence in the answer only in percent (0–100 %) direcly, then following the rationales about your response:
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def compute_rate(avg_logprobs, calibrate_data):
    return (np.mean(np.array(calibrate_data) <= avg_logprobs) * 100).round(2)


def load_calibrate_data():
    logging.info("Loading calibrate data")
    calibrate_data = []
    for dataset_name in ["gsm8k", "medqa_us", "triviaqa"]:
        with open(
            f"benchmark/results/Qwen2.5-7B-Instruct/{dataset_name}/calibration.json",
            "r",
        ) as f:
            calibrate_data.extend(json.load(f))
    return calibrate_data


calibrate_data = load_calibrate_data()


@function
def qa(s, messages: List[Dict[str, str]]):
    for message in messages:
        if message["role"] == "user":
            s += user(message["content"])
        elif message["role"] == "assistant":
            s += assistant(message["content"])
    s += assistant(gen(name="rationale", return_logprob=True))

    # print(s.get_meta_info("rationale"))
    avg_logprobs = np.mean(
        [tok[0] for tok in s.get_meta_info("rationale")["output_token_logprobs"]]
    )
    rate = compute_rate(avg_logprobs, calibrate_data)
    s += user(second_prompt.format(rate=rate))
    s += assistant(gen(name="confidence"))


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
    gr.Markdown("## Uncertainty Quantification", elem_id="notice_markdown")

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

    state = qa.run(
        messages=history_state,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        stream=True,
    )
    history_state.append({"role": "assistant", "content": ""})
    for new_text in state.text_iter(var_name="rationale"):
        history_state[-1]["content"] += new_text
        yield history_state, history_state, disable_btn, disable_btn, disable_btn, disable_btn

    history_state[-1]["content"] += "\n\n"
    for new_text in state.text_iter(var_name="confidence"):
        history_state[-1]["content"] += new_text
        yield history_state, history_state, disable_btn, disable_btn, disable_btn, disable_btn
    yield history_state, history_state, enable_btn, enable_btn, enable_btn, enable_btn


def build_demo():
    text_size = gr.themes.sizes.text_lg
    with gr.Blocks(
        title="Qwen2.5-7B-Instruct - Uncertainty Quantification",
        theme=gr.themes.Default(text_size=text_size),
    ) as demo:
        with gr.Tabs() as inner_tabs:
            with gr.Tab("💬 Direct Chat", id=2) as direct_tab:
                direct_tab.select(None, None, None)
                build_single_model_ui()
    return demo


@hydra.main(config_path=".", config_name="serve", version_base="1.2")
def main(config: DictConfig):
    logging.info(f"tp_size: {config.tp_size}, dp_size: {config.dp_size}")
    set_default_backend(
        Runtime(
            model_path="Qwen/Qwen2.5-7B-Instruct",
            tp_size=config.tp_size,
            dp_size=config.dp_size,
            dtype="bfloat16",
        )
    )

    demo = build_demo()
    demo.launch(
        share=config.share,
        server_name=config.server_name,
        server_port=config.server_port,
    )


if __name__ == "__main__":
    main()
