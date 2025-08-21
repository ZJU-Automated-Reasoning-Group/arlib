"""Validation utilities for MCP solver integration."""

import ast
import re
from datetime import timedelta
from typing import Any
from arlib.utils.exceptions import ArlibException

class ValidationError(ArlibException):
    """Exception for validation errors."""
    pass

# Constants
MAX_CODE_LENGTH = 100000
MIN_SOLVE_TIMEOUT = timedelta(seconds=1)
MAX_SOLVE_TIMEOUT = timedelta(seconds=30)

def validate_content(content: Any) -> None:
    """Validate model item content."""
    if not isinstance(content, str) or not content.strip():
        raise ValidationError("Content must be a non-empty string")
    if len(content) > MAX_CODE_LENGTH:
        raise ValidationError(f"Content too long: {len(content)} > {MAX_CODE_LENGTH}")

def validate_python_code_safety(code: str) -> None:
    """Validate Python code safety."""
    dangerous = [r"\bexec\s*\(", r"\beval\s*\(", r"\bopen\s*\(", r"\bimport\s+os\b", r"__import__\s*\("]

    try:
        ast.parse(code)
    except SyntaxError as e:
        raise ValidationError(f"Syntax error: {e}") from e

    for pattern in dangerous:
        if re.search(pattern, code):
            raise ValidationError("Unsafe code pattern detected")

def validate_timeout(timeout: Any, min_t: timedelta, max_t: timedelta) -> None:
    """Validate timeout duration."""
    if not isinstance(timeout, timedelta) or timeout < min_t or timeout > max_t:
        raise ValidationError(f"Invalid timeout: must be between {min_t.total_seconds()}s and {max_t.total_seconds()}s")

def get_standardized_response(success: bool, message: str, error: str | None = None, **kwargs) -> dict[str, Any]:
    """Create standardized response."""
    response = {"success": success if not error else False, "message": message}
    if error:
        response.update({"error": error, "status": "error"})
    response.update(kwargs)
    return response
