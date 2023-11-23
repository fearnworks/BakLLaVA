import argparse

import os

import gradio as gr


from llava.utils import build_logger
from dotenv import load_dotenv
from .about import block_css, learn_more_markdown, title_markdown, tos_markdown

load_dotenv()
from .config import *
from .chat import *
from .model import *
from .prompt_templates import load_prompt_values, register_config_handlers

logger = build_logger("gradio_web_server", "logs/gradio_web_server.log")


def build_demo():
    saved_values = load_prompt_values()

    textbox = gr.Textbox(
        show_label=False, placeholder="Enter text and press ENTER", container=False
    )
    with gr.Blocks(title="LLaVA", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("Config"):
                        with gr.Row(elem_id="model_selector_row"):
                            model_selector = gr.Dropdown(
                                choices=models,
                                value=models[0] if len(models) > 0 else "",
                                interactive=True,
                                show_label=False,
                                container=False,
                            )

                        imagebox = gr.Image(type="pil")
                        image_process_mode = gr.Radio(
                            ["Crop", "Resize", "Pad", "Default"],
                            value="Default",
                            label="Preprocess for non-square image",
                            visible=True,
                        )

                        save_button = gr.Button("Save Prompts")
                        load_button = gr.Button("Load Prompts")

                        with gr.Accordion("Examples", open=False):
                            cur_dir = os.path.dirname(os.path.abspath(__file__))
                            gr.Examples(
                                examples=[
                                    [
                                        f"{cur_dir}/examples/extreme_ironing.jpg",
                                        "What is unusual about this image?",
                                    ],
                                    [
                                        f"{cur_dir}/examples/waterview.jpg",
                                        "What are the things I should be cautious about when I visit here?",
                                    ],
                                ],
                                inputs=[imagebox, textbox],
                            )
                        with gr.Accordion("About", open=False):
                            gr.Markdown(title_markdown)
                            gr.Markdown(tos_markdown)
                            gr.Markdown(learn_more_markdown)
                            url_params = gr.JSON(visible=True)

                    with gr.Tab("Parameters"):
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.2,
                            step=0.1,
                            interactive=True,
                            label="Temperature",
                        )
                        top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.7,
                            step=0.1,
                            interactive=True,
                            label="Top P",
                        )
                        max_output_tokens = gr.Slider(
                            minimum=0,
                            maximum=1024,
                            value=512,
                            step=64,
                            interactive=True,
                            label="Max output tokens",
                        )
                    with gr.Tab("Responses"):
                        with gr.Column():  # Use a single column for this tab's content
                            # Textboxes area
                            predefined_response_1 = gr.Textbox(
                                label="Response 1",
                                value=saved_values.get("response1", ""),
                                placeholder="Enter response 1",
                            )
                            predefined_response_2 = gr.Textbox(
                                label="Response 2",
                                value=saved_values.get("response2", ""),
                                placeholder="Enter response 2",
                            )
                            predefined_response_3 = gr.Textbox(
                                label="Response 3",
                                value=saved_values.get("response3", ""),
                                placeholder="Enter response 3",
                            )

                            # Buttons area
                            with gr.Row():  # Use a row to align buttons horizontally
                                predefined_submit_1 = gr.Button(value="Submit 1")
                                predefined_submit_2 = gr.Button(value="Submit 2")
                                predefined_submit_3 = gr.Button(value="Submit 3")

                        with gr.Row():
                            system_prompt = gr.Textbox(
                                label="System Prompt",
                                value=saved_values.get("system_prompt", ""),
                                placeholder="Enter system prompt here",
                                container=False,
                            )
                        use_system_prompt = gr.Checkbox(
                            label="Use System Prompt", value=True
                        )

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot", label="LLaVA Chatbot", height=550
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        chat_action_buttons = {
            'upvote_btn': upvote_btn,
            'downvote_btn': downvote_btn,
            'flag_btn': flag_btn,
            'regenerate_btn': regenerate_btn,
            'clear_btn': clear_btn
        }

        register_chat_actionbar_handlers(
            chat_buttons=chat_action_buttons,
            state=state,
            model_selector=model_selector,
            image_process_mode=image_process_mode,
            textbox=textbox,
            chatbot=chatbot,
            imagebox=imagebox
        )
        #############################
        interaction_elements = {
            'textbox': textbox,
            'submit_btn': submit_btn,
            'predefined_submit_1': predefined_submit_1,
            'predefined_submit_2': predefined_submit_2,
            'predefined_submit_3': predefined_submit_3,
            'predefined_response_1': predefined_response_1,
            'predefined_response_2': predefined_response_2,
            'predefined_response_3': predefined_response_3,
            'use_system_prompt': use_system_prompt,
            'system_prompt': system_prompt,
            'model_selector': model_selector,
            'temperature': temperature,
            'top_p': top_p,
            'max_output_tokens': max_output_tokens,
            'btn_list': btn_list
        }
        # Register response handlers
        register_response_handlers(
            interaction_elements=interaction_elements,
            state=state,
            chatbot=chatbot,
            imagebox=imagebox,
            image_process_mode=image_process_mode
        )
        
        ### Register config
        config_buttons = {
            "save_button": save_button,
            "load_button": load_button
        }

        # List of predefined response textboxes
        predefined_responses = [
            predefined_response_1,
            predefined_response_2,
            predefined_response_3
        ]
        register_config_handlers(
            config_buttons=config_buttons,
            state=state,
            predefined_responses=predefined_responses,
            system_prompt=system_prompt
        )
        ######################
        if args.model_list_mode == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                models,
                _js=get_window_url_params,
            )
        elif args.model_list_mode == "reload":
            demo.load(load_demo_refresh_model_list, None, [state, model_selector])
        else:
            raise ValueError(f"Unknown model list mode: {args.model_list_mode}")

    return demo


if __name__ == "__main__":
    logger.info(f"args: {args}")

    models = get_model_list()

    logger.info(args)
    demo = build_demo()
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(server_name=args.host, server_port=args.port, share=args.share)
