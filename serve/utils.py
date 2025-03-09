import datetime
import json
import logging
import os
import time

import gradio as gr
import torch.nn.functional as F

LOGDIR = os.getenv("LOGDIR", "logs")
os.makedirs(LOGDIR, exist_ok=True)
no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True, visible=True)
disable_btn = gr.Button(interactive=False)
invisible_btn = gr.Button(interactive=False, visible=False)


def flag_last_response(state, request: gr.Request):
    ip = get_ip(request)
    logging.info(f"flag. ip: {ip}")
    vote_last_response(state, "flag", request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, request: gr.Request):
    ip = get_ip(request)
    logging.info(f"regenerate. ip: {ip}")
    last_conv = state[-2]
    state = state[:-2]

    return (state, state, last_conv["content"], "") + (disable_btn,) * 4


def clear_history(request: gr.Request):
    ip = get_ip(request)
    logging.info(f"clear_history. ip: {ip}")
    return ([], [], "") + (disable_btn,) * 4


def get_ip(request: gr.Request):
    if "cf-connecting-ip" in request.headers:
        ip = request.headers["cf-connecting-ip"]
    elif "x-forwarded-for" in request.headers:
        ip = request.headers["x-forwarded-for"]
        if "," in ip:
            ip = ip.split(",")[0]
    else:
        ip = request.client.host
    return ip


def downvote_last_response(state, request: gr.Request):
    ip = get_ip(request)
    logging.info(f"downvote. ip: {ip}")
    vote_last_response(state, "downvote", request)
    return ("",) + (disable_btn,) * 2


def upvote_last_response(state, request: gr.Request):
    ip = get_ip(request)
    logging.info(f"upvote. ip: {ip}")
    vote_last_response(state, "upvote", request)
    return ("",) + (disable_btn,) * 2


def get_conv_log_filename():
    t = datetime.datetime.now()
    conv_log_filename = f"{t.year}-{t.month:02d}-{t.day:02d}-conv.jsonl"
    name = os.path.join(LOGDIR, conv_log_filename)
    return name


def vote_last_response(state, vote_type, request: gr.Request):
    filename = get_conv_log_filename()

    with open(filename, "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "state": state[-2:],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")


def compute_log_prob(model, input_ids, assistant_length: int):
    labels = input_ids[0, -assistant_length:]
    outputs = model(input_ids)
    logits = outputs.logits[0, -assistant_length - 1 : -1]
    probs = F.log_softmax(logits, dim=-1)
    avg_logprob = probs[range(len(labels)), labels].mean().item()
    return avg_logprob
