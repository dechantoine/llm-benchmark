# This file contains mixin functions for the llm module.

import re

def format_preprompt(preprompt: str, labels: list[str], with_index: bool = False) -> str:
    if with_index:
        labels = [f"{i + 1}. {label}" for i, label in enumerate(labels)]
    return f"""{preprompt}[{', '.join(labels)}]"""