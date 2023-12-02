import os
import sys
from loguru import logger 
import requests

from llava.constants import LOGDIR

server_error_msg = (
    "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
)
moderation_msg = (
    "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."
)

handler = None


def build_logger(logger_name: str, logger_filename: str, output_dir: str = LOGDIR):
    """
    Builds a loguru logger with a file sink.

    Args:
        logger_name (str): Name of the logger.
        logger_filename (str): Filename for the log file.

    Returns:
        logger: Configured loguru logger instance.
    """
    # Set up the log directory
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, logger_filename)

    # Configure loguru logger
    logger.remove()
    logger.add(
        sys.stdout, colorize=True, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | {message}"
    )
    logger.add(
        filename, rotation="1 day", level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"
    )

    # Configure logger name
    local_logger = logger.bind(name=logger_name)

    return local_logger

class StreamToLogger:
    """
    Redirects writes to a loguru logger instance.

    Args:
        log_level (str): Log level for messages written to this stream.
    """

    def __init__(self, log_level="INFO"):
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass



def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)



def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
    }
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"
