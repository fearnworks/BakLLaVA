from typing import Dict, List, Any, Tuple
from gradio import Button, State, Textbox, Image, Checkbox, Radio, Chatbot

# prompt_templates.py
import json
import os
from loguru import logger
from .chat import add_text
from .config import *


def register_config_handlers(
    config_buttons: Dict[str, Button],
    state: State,
    predefined_responses: List[Textbox],
    system_prompt: Textbox,
):
    """
    Registers handlers for configuration-related buttons.

    Args:
        config_buttons: A dictionary of button components for config actions.
        state: The shared state object for the Gradio interface.
        predefined_responses: A list of Textbox components holding predefined responses.
        system_prompt: The Textbox component for the system prompt.
    """
    config_buttons["save_button"].click(
        handle_save, [state] + predefined_responses + [system_prompt]
    )
    config_buttons["load_button"].click(
        handle_load, [state], predefined_responses + [system_prompt]
    )


# Exported functions to interact with saved prompts
def handle_save(
    state: State, response1: str, response2: str, response3: str, system_prompt: str
) -> str:
    """
    Saves the provided prompt values to a file.

    Args:
        state: The shared state object for the Gradio interface.
        response1: The first predefined response.
        response2: The second predefined response.
        response3: The third predefined response.
        system_prompt: The system prompt text.

    Returns:
        A success message indicating that the prompts have been saved.
    """
    logger.info("Saving Prompts")
    prompt_values = {
        "response1": response1,
        "response2": response2,
        "response3": response3,
        "system_prompt": system_prompt,
    }
    logger.info(f"Prompt Values: {prompt_values}")
    save_prompt_values(prompt_values)
    return "Prompts saved successfully."


def handle_load(state: State) -> Tuple[str, str, str, str]:
    """
    Loads the saved prompt values from a file.

    Args:
        state: The shared state object for the Gradio interface.

    Returns:
        A tuple containing the loaded prompt values.
    """
    logger.info("Loading Prompts")
    prompt_values = load_prompt_values()
    return (
        prompt_values["response1"],
        prompt_values["response2"],
        prompt_values["response3"],
        prompt_values["system_prompt"],
    )


# Define functions to handle prompt values
def save_prompt_values(prompt_values: Dict[str, str]):
    """
    Saves the given prompt values to a file.

    Args:
        prompt_values: A dictionary containing the prompt values to save.
    """
    with open(prompt_values_file, "w") as file:
        json.dump(prompt_values, file)


def load_prompt_values() -> Dict[str, str]:
    """
    Loads the prompt values from a file.

    Returns:
        A dictionary containing the prompt values.
    """
    if os.path.exists(prompt_values_file):
        with open(prompt_values_file, "r") as file:
            return json.load(file)
    else:
        return {"response1": "", "response2": "", "response3": "", "system_prompt": ""}
