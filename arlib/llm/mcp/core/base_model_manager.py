"""Base model manager implementation."""

from datetime import timedelta
from typing import Any
from .base_manager import SolverManager

class BaseModelManager(SolverManager):
    """Base implementation with common model management functionality."""

    def __init__(self):
        super().__init__()
        self.code_items: list[str] = []
        self.last_result = None
        self.last_solution = None
        self.last_solve_time = None

    async def clear_model(self) -> dict[str, Any]:
        """Clear the current model."""
        self.code_items.clear()
        self.last_result = self.last_solution = self.last_solve_time = None
        return {"success": True, "message": "Model cleared"}

    async def add_item(self, index: int, content: str) -> dict[str, Any]:
        """Add item at index."""
        if not (0 <= index <= len(self.code_items)):
            return {"success": False, "error": f"Invalid index {index}"}
        self.code_items.insert(index, content)
        return {"success": True, "message": f"Added item at index {index}"}

    async def delete_item(self, index: int) -> dict[str, Any]:
        """Delete item at index."""
        if not (0 <= index < len(self.code_items)):
            return {"success": False, "error": f"Invalid index {index}"}
        del self.code_items[index]
        return {"success": True, "message": f"Deleted item at index {index}"}

    async def replace_item(self, index: int, content: str) -> dict[str, Any]:
        """Replace item at index."""
        if not (0 <= index < len(self.code_items)):
            return {"success": False, "error": f"Invalid index {index}"}
        self.code_items[index] = content
        return {"success": True, "message": f"Replaced item at index {index}"}

    def get_model(self) -> list[tuple[int, str]]:
        """Get current model."""
        return list(enumerate(self.code_items))

    def _get_full_code(self) -> str:
        """Get full code."""
        return "\n\n".join(self.code_items)

    def get_solution(self) -> dict[str, Any]:
        """Get current solution."""
        return self.last_solution or {"error": "No solution available"}

    def get_variable_value(self, variable_name: str) -> dict[str, Any]:
        """Get variable value."""
        if not self.last_solution:
            return {"error": "No solution available"}
        values = self.last_solution.get("values", {})
        return {"variable": variable_name, "value": values.get(variable_name)} if variable_name in values else {"error": f"Variable '{variable_name}' not found"}

    def get_solve_time(self) -> dict[str, Any]:
        """Get solve time."""
        return {"solve_time": self.last_solve_time, "unit": "seconds"} if self.last_solve_time else {"error": "No solve operation performed"}
