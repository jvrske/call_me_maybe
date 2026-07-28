from .models import FunctionDefinition, PromptInput
import json


def load_func_defs(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            js = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found in {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")
    return [FunctionDefinition(**item) for item in js]


def load_prompts(path):
    try:
        