"""
Abstract base class for solver managers.
"""

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any


class SolverManager(ABC):
    """
    Abstract base class for solver managers.
    This class defines the interface that all solver implementations must follow.
    """

    def __init__(self):
        """Initialize a new solver manager."""
        self.initialized = False
        self.last_solve_time = None

    @abstractmethod
    async def clear_model(self) -> dict[str, Any]:
        """Clear the current model."""
        pass

    @abstractmethod
    async def add_item(self, index: int, content: str) -> dict[str, Any]:
        """Add an item to the model at the specified index."""
        pass

    @abstractmethod
    async def delete_item(self, index: int) -> dict[str, Any]:
        """Delete an item from the model at the specified index."""
        pass

    @abstractmethod
    async def replace_item(self, index: int, content: str) -> dict[str, Any]:
        """Replace an item in the model at the specified index."""
        pass

    @abstractmethod
    async def solve_model(self, timeout: timedelta) -> dict[str, Any]:
        """Solve the current model."""
        pass

    @abstractmethod
    def get_solution(self) -> dict[str, Any]:
        """Get the current solution."""
        pass

    @abstractmethod
    def get_variable_value(self, variable_name: str) -> dict[str, Any]:
        """Get the value of a variable from the current solution."""
        pass

    @abstractmethod
    def get_solve_time(self) -> dict[str, Any]:
        """Get the time taken for the last solve operation."""
        pass
