"""
Z3 model manager implementation using BaseModelManager.
"""

import logging
from datetime import timedelta
from typing import Any

import z3

from ..core.base_model_manager import BaseModelManager
from ..core.validation import (
    ValidationError,
    get_standardized_response,
    validate_content,
    validate_python_code_safety,
    validate_timeout,
    MIN_SOLVE_TIMEOUT,
    MAX_SOLVE_TIMEOUT,
)
from .environment import execute_z3_code


class Z3ModelManager(BaseModelManager):
    """Z3 model manager implementation."""

    def __init__(self):
        """Initialize a new Z3 model manager."""
        super().__init__()
        self.initialized = True
        self._registry = {"variables": {}, "solver": None}
        self.logger = logging.getLogger(__name__)
        self.logger.info("Z3 model manager initialized")

    async def clear_model(self) -> dict[str, Any]:
        """Clear the current model."""
        result = await super().clear_model()
        self._registry = {"variables": {}, "solver": None}
        self.logger.info("Model cleared")
        return get_standardized_response(success=True, message="Model cleared")

    async def add_item(self, index: int, content: str) -> dict[str, Any]:
        """Add an item to the model at the specified index."""
        try:
            validate_content(content)
            validate_python_code_safety(content)

            result = await super().add_item(index, content)
            if not result.get("success"):
                return result

            model = self.get_model()
            self.logger.info(f"Added item at index {index}")
            return get_standardized_response(
                success=True,
                message=f"Added item at index {index}",
                model=model,
            )

        except ValidationError as e:
            error_msg = str(e)
            self.logger.error(f"Validation error in add_item: {error_msg}")
            return get_standardized_response(
                success=False,
                message=f"Failed to add item: {error_msg}",
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Unexpected error in add_item: {e!s}"
            self.logger.error(error_msg, exc_info=True)
            return get_standardized_response(
                success=False,
                message="Failed to add item due to an internal error",
                error=error_msg,
            )

    async def delete_item(self, index: int) -> dict[str, Any]:
        """Delete an item from the model at the specified index."""
        result = await super().delete_item(index)
        if not result.get("success"):
            return result

        model = self.get_model()
        self.logger.info(f"Deleted item at index {index}")
        return get_standardized_response(
            success=True,
            message=f"Deleted item at index {index}",
            model=model,
        )

    async def replace_item(self, index: int, content: str) -> dict[str, Any]:
        """Replace an item in the model at the specified index."""
        try:
            validate_content(content)
            validate_python_code_safety(content)

            result = await super().replace_item(index, content)
            if not result.get("success"):
                return result

            model = self.get_model()
            self.logger.info(f"Replaced item at index {index}")
            return get_standardized_response(
                success=True,
                message=f"Replaced item at index {index}",
                model=model,
            )

        except ValidationError as e:
            error_msg = str(e)
            self.logger.error(f"Validation error in replace_item: {error_msg}")
            model = self.get_model()
            return get_standardized_response(
                success=False,
                message=f"Failed to replace item: {error_msg}",
                error=error_msg,
                model=model,
            )
        except Exception as e:
            error_msg = f"Unexpected error in replace_item: {e!s}"
            self.logger.error(error_msg, exc_info=True)
            model = self.get_model()
            return get_standardized_response(
                success=False,
                message="Failed to replace item due to an internal error",
                error=error_msg,
                model=model,
            )

    async def solve_model(self, timeout: timedelta) -> dict[str, Any]:
        """Solve the current model."""
        try:
            if not self.code_items:
                return get_standardized_response(
                    success=False, message="Model is empty", error="Empty model"
                )

            validate_timeout(timeout, MIN_SOLVE_TIMEOUT, MAX_SOLVE_TIMEOUT)

            combined_code = self._get_full_code()

            # The code is ready to execute with export_solution available

            timeout_seconds = timeout.total_seconds()
            result = execute_z3_code(combined_code, timeout=timeout_seconds)

            self.last_result = result
            self.last_solve_time = result.get("execution_time")

            # The solution should be captured by export_solution in the executed code

            # Store solution
            if result.get("solution"):
                self.last_solution = result.get("solution")

            # Format result for client
            formatted_result = {
                "success": True,
                "message": "Model solved",
                "status": result.get("status", "unknown"),
                "output": result.get("output", []),
                "execution_time": result.get("execution_time", 0),
            }

            if result.get("error"):
                formatted_result["error"] = result.get("error")
                formatted_result["success"] = False
                formatted_result["message"] = "Failed to solve model"

            if result.get("solution"):
                solution = result.get("solution", {})
                formatted_result["satisfiable"] = solution.get("satisfiable", False)
                formatted_result["values"] = solution.get("values", {})

                if solution.get("objective") is not None:
                    formatted_result["objective"] = solution.get("objective")

                if solution.get("output") and isinstance(solution.get("output"), list):
                    formatted_result["output"].extend(solution.get("output"))

                if "property_verified" in solution.get("values", {}):
                    property_verified = solution["values"]["property_verified"]
                    formatted_result["property_verified"] = property_verified

                    if property_verified:
                        if not any("verified" in line.lower() for line in formatted_result["output"]):
                            formatted_result["output"].append("Property verified successfully.")
                    else:
                        if not any("counterexample" in line.lower() for line in formatted_result["output"]):
                            formatted_result["output"].append("Property verification failed. Counterexample found.")

            return formatted_result

        except ValidationError as e:
            error_msg = str(e)
            self.logger.error(f"Validation error in solve_model: {error_msg}")
            return get_standardized_response(
                success=False,
                message=f"Failed to solve model: {error_msg}",
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Unexpected error in solve_model: {e!s}"
            self.logger.error(error_msg, exc_info=True)
            return get_standardized_response(
                success=False,
                message="Failed to solve model due to an internal error",
                error=error_msg,
            )

    def get_solution(self) -> dict[str, Any]:
        """Get the current solution."""
        if not self.last_solution:
            return get_standardized_response(
                success=False, message="No solution available", error="No solution"
            )

        return get_standardized_response(
            success=True,
            message="Solution retrieved",
            satisfiable=self.last_solution.get("satisfiable", False),
            values=self.last_solution.get("values", {}),
            objective=self.last_solution.get("objective"),
            status=self.last_solution.get("status", "unknown"),
        )

    def get_variable_value(self, variable_name: str) -> dict[str, Any]:
        """Get the value of a variable from the current solution."""
        if not self.last_solution:
            return get_standardized_response(
                success=False, message="No solution available", error="No solution"
            )

        values = self.last_solution.get("values", {})

        if variable_name not in values:
            return get_standardized_response(
                success=False,
                message=f"Variable '{variable_name}' not found in solution",
                error="Variable not found",
            )

        return get_standardized_response(
            success=True,
            message=f"Value of variable '{variable_name}'",
            value=values.get(variable_name),
        )

    def get_solve_time(self) -> dict[str, Any]:
        """Get the time taken for the last solve operation."""
        if self.last_solve_time is None:
            return get_standardized_response(
                success=False,
                message="No solve operation has been performed",
                error="No solve time available",
            )

        return get_standardized_response(
            success=True,
            message="Solve time retrieved",
            solve_time=self.last_solve_time,
        )
