from loguru import logger
import gradio as gr
import json
from llava.conversation import default_conversation, conv_templates, SeparatorStyle
from llava.utils import server_error_msg
import datetime
import time
import os
from llava.constants import LOGDIR
import hashlib
import requests
from .config import args
from typing import Dict
from gradio import Button, State, Textbox, Chatbot, Image, Radio, Request

headers = {"User-Agent": "LLaVA Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def add_text(
    state,
    use_system_prompt,
    system_prompt,
    text,
    image,
    image_process_mode,
    request: gr.Request,
):
    if use_system_prompt:
        text = system_prompt + text
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 5
    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if "<image>" not in text:
            # text = '<Image><image></Image>' + text
            text = text + "\n<image>"
        text = (text, image, image_process_mode)
        if len(state.get_images(return_pil=True)) > 0:
            state = default_conversation.copy()
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 5


def http_bot(
    state, model_selector, temperature, top_p, max_new_tokens, request: gr.Request
):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector

    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "llava" in model_name.lower():
            if "llama-2" in model_name.lower():
                template_name = "llava_llama_2"
            elif "v1" in model_name.lower():
                if "mmtag" in model_name.lower():
                    template_name = "v1_mmtag"
                elif (
                    "plain" in model_name.lower()
                    and "finetune" not in model_name.lower()
                ):
                    template_name = "v1_mmtag"
                else:
                    template_name = "llava_v1"
            elif "mpt" in model_name.lower():
                template_name = "mpt"
            else:
                if "mmtag" in model_name.lower():
                    template_name = "v0_mmtag"
                elif (
                    "plain" in model_name.lower()
                    and "finetune" not in model_name.lower()
                ):
                    template_name = "v0_mmtag"
                else:
                    template_name = "llava_v0"
        elif "mpt" in model_name:
            template_name = "mpt_text"
        elif "llama-2" in model_name:
            template_name = "llama_2"
        else:
            template_name = "vicuna_v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (
            state,
            state.to_gradio_chatbot(),
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    # Construct prompt
    prompt = state.get_prompt()

    all_images = state.get_images(return_pil=True)
    all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
    for image, hash in zip(all_images, all_image_hash):
        t = datetime.datetime.now()
        filename = os.path.join(
            LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg"
        )
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

    # Make requests
    pload = {
        "model": model_name,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": state.sep
        if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT]
        else state.sep2,
        "images": f"List of {len(state.get_images())} images: {all_image_hash}",
    }
    logger.info(f"==== request ====\n{pload}")

    pload["images"] = state.get_images()

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5

    try:
        # Stream output
        response = requests.post(
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=pload,
            stream=True,
            timeout=10,
        )
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(prompt) :].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                    )
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(),
            "images": all_image_hash,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def register_chat_actionbar_handlers(
    chat_buttons: Dict[str, Button],
    state: State,
    model_selector: Textbox,
    image_process_mode: Radio,
    textbox: Textbox,
    chatbot: Chatbot,
    imagebox: Image
):
    """
    Registers handlers for chat-related buttons.

    Args:
        chat_buttons: A dictionary of button components for chat actions.
        state: The shared state object for the Gradio interface.
        model_selector: The Textbox component for selecting the model.
        image_process_mode: Radio component to select image processing mode.
        textbox: Textbox component for user input.
        chatbot: Chatbot component for conversation display.
        imagebox: Image component for image input.
    """
    chat_buttons['upvote_btn'].click(
        upvote_last_response, 
        [state, model_selector], 
        [textbox]
    )
    chat_buttons['downvote_btn'].click(
        downvote_last_response, 
        [state, model_selector], 
        [textbox]
    )
    chat_buttons['flag_btn'].click(
        flag_last_response, 
        [state, model_selector], 
        [textbox]
    )
    chat_buttons['regenerate_btn'].click(
        regenerate, 
        [state, image_process_mode], 
        [state, chatbot, textbox, imagebox]
    )
    chat_buttons['clear_btn'].click(
        clear_history, 
        None, 
        [state, chatbot, textbox, imagebox]
    )
    
def register_response_handlers(interaction_elements, state, chatbot, imagebox, image_process_mode):
    """
    Registers handlers for user interaction elements including textbox submissions and button clicks.

    Args:
        interaction_elements: A dictionary containing all the interactive elements and their buttons.
        state: The shared state object for the Gradio interface.
        chatbot: Chatbot component for displaying conversation.
        imagebox: Image component for image input.
        image_process_mode: Radio component for selecting image processing mode.
    """
    # Register the main textbox submission handler
    interaction_elements['textbox'].submit(
        add_text,
        [
            state,
            interaction_elements['use_system_prompt'],
            interaction_elements['system_prompt'],
            interaction_elements['textbox'],
            imagebox,
            image_process_mode,
        ],
        [state, chatbot, interaction_elements['textbox'], imagebox] + interaction_elements['btn_list'],
    ).then(
        http_bot,
        [state, interaction_elements['model_selector'], interaction_elements['temperature'], interaction_elements['top_p'], interaction_elements['max_output_tokens']],
        [state, chatbot] + interaction_elements['btn_list'],
    )

    # Register handlers for the submit buttons
    submit_buttons = ['submit_btn', 'predefined_submit_1', 'predefined_submit_2', 'predefined_submit_3']
    for submit_btn_key in submit_buttons:
        response_key = submit_btn_key.replace('submit', 'response') if 'predefined' in submit_btn_key else 'textbox'
        interaction_elements[submit_btn_key].click(
            add_text,
            [
                state,
                interaction_elements['use_system_prompt'],
                interaction_elements['system_prompt'],
                interaction_elements[response_key],
                imagebox,
                image_process_mode,
            ],
            [state, chatbot, interaction_elements['textbox'], imagebox] + interaction_elements['btn_list'],
        ).then(
            http_bot,
            [state, interaction_elements['model_selector'], interaction_elements['temperature'], interaction_elements['top_p'], interaction_elements['max_output_tokens']],
            [state, chatbot],
        )