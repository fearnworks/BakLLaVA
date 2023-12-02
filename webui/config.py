# config.py
import argparse
import os
from loguru import logger

class AppConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._parse_args(cls._instance)
        return cls._instance

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", type=str, default="0.0.0.0")
        parser.add_argument("--port", type=int)
        parser.add_argument(
            "--controller-url", type=str, default="http://localhost:21001"
        )
        parser.add_argument("--concurrency-count", type=int, default=8)
        parser.add_argument(
            "--model-list-mode", type=str, default="once", choices=["once", "reload"]
        )
        parser.add_argument("--share", action="store_true")
        parser.add_argument("--moderate", action="store_true")
        parser.add_argument("--embed", action="store_true")
        self.args = parser.parse_args()


# Set up global config variables
output_directory = os.getenv(
    "OUTPUT_DIR", "./"
)  # Default to current directory if OUTPUT_DIR not set
prompt_values_file = os.path.join(output_directory, "prompt_values.json")
priority = {
    "vicuna-13b": "aaaaaaa",
    "koala-13b": "aaaaaab",
}
headers = {"User-Agent": "LLaVA Client"}


# Define functions to handle CLI arguments
def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=8)
    parser.add_argument(
        "--model-list-mode", type=str, default="once", choices=["once", "reload"]
    )
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    return parser.parse_args()


config = AppConfig()
args = config.args
