"""
Prompt loading utilities for MCP solver integration.
"""

import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Type definitions
PromptMode = Literal["z3"]
PromptType = Literal["instructions", "review"]


def get_prompt_path(mode: PromptMode, prompt_type: PromptType = "instructions") -> Path:
    """
    Get the path to a prompt file based on mode and type.

    Args:
        mode: The solver mode (currently only "z3")
        prompt_type: The type of prompt ("instructions" or "review")

    Returns:
        Path object pointing to the prompt file

    Raises:
        ValueError: If invalid mode or prompt type is provided
    """
    if mode not in ("z3",):
        raise ValueError(f"Invalid mode: {mode}. Must be: z3")

    if prompt_type not in ("instructions", "review"):
        raise ValueError(
            f"Invalid prompt type: {prompt_type}. Must be one of: instructions, review"
        )

    # Get the prompts directory path
    base_path = Path(__file__).parent

    # Construct the full path to the prompt file
    prompt_path = base_path / f"{mode}_{prompt_type}.md"

    logger.debug(f"Prompt path for {prompt_type} in {mode} mode: {prompt_path}")
    return prompt_path


def load_prompt(mode: PromptMode, prompt_type: PromptType) -> str:
    """
    Load a prompt file based on mode and type.

    Args:
        mode: The solver mode (currently only "z3")
        prompt_type: The type of prompt ("instructions" or "review")

    Returns:
        The content of the prompt file as a string

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        ValueError: If invalid mode or prompt type is provided
    """
    prompt_path = get_prompt_path(mode, prompt_type)

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    logger.debug(f"Loading {prompt_type} prompt for {mode} mode from: {prompt_path}")

    try:
        content = prompt_path.read_text(encoding="utf-8").strip()
        logger.debug(f"Successfully loaded prompt ({len(content)} characters)")
        return content
    except Exception as e:
        raise RuntimeError(f"Error reading prompt file {prompt_path}: {e!s}")
