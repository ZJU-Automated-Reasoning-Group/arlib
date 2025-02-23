"""
Configuration manager for SMT and SAT solvers.
Provides a singleton class to manage solver paths and availability.
"""

from pathlib import Path
import shutil
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SolverConfig:
    """Configuration details for a single solver"""

    def __init__(self, name: str, exec_name: str):
        self.name = name
        self.exec_name = exec_name
        self.exec_path: Optional[str] = None
        self.version: Optional[str] = None
        self.is_available: bool = False


class SolverRegistry(type):
    """Metaclass for singleton pattern"""
    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__call__(*args, **kwargs)
        return cls._instance


class GlobalConfig(metaclass=SolverRegistry):
    """
    Global configuration manager for solver executables.
    Implements singleton pattern to ensure consistent solver configurations.
    """

    # Default solver configurations
    SOLVERS = {
        "z3": SolverConfig("z3", "z3"),
        "cvc5": SolverConfig("cvc5", "cvc5"),
        "mathsat": SolverConfig("mathsat", "mathsat"),
        "yices2": SolverConfig("yices2", "yices-smt2"),
        "sharp_sat": SolverConfig("sharp_sat", "sharpSAT")
    }

    def __init__(self):
        """Initialize solver configurations and locate executables"""
        self._bin_solver_path = Path(__file__).parent.parent.parent / "bin_solvers"
        self._locate_all_solvers()

    def _locate_solver(self, solver_config: SolverConfig) -> None:
        """
        Locate a solver executable and update its configuration.

        Args:
            solver_config: SolverConfig object to update
        """

        # Try bin_solvers directory
        local_path = self._bin_solver_path / solver_config.exec_name
        if shutil.which(str(local_path)):
            solver_config.exec_path = str(local_path)
            solver_config.is_available = True
            return

        # Try system PATH
        system_path = shutil.which(solver_config.exec_name)
        if system_path:
            solver_config.exec_path = system_path
            solver_config.is_available = True
            return

        logger.warning(f"Could not locate {solver_config.name} solver executable")

    def _locate_all_solvers(self) -> None:
        """Locate all configured solvers"""
        for solver_config in self.SOLVERS.values():
            self._locate_solver(solver_config)

    def set_solver_path(self, solver_name: str, path: str) -> None:
        """
        Set a custom path for a solver executable.

        Args:
            solver_name: Name of the solver
            path: Custom path to the solver executable

        Raises:
            ValueError: If solver name is invalid or path doesn't exist
        """
        if solver_name not in self.SOLVERS:
            raise ValueError(f"Unknown solver: {solver_name}")

        if not Path(path).exists():
            raise ValueError(f"Path does not exist: {path}")

        self._locate_solver(self.SOLVERS[solver_name])

    def get_solver_path(self, solver_name: str) -> Optional[str]:
        """
        Get the path to a solver executable.

        Args:
            solver_name: Name of the solver

        Returns:
            Path to the solver executable or None if not available

        Raises:
            ValueError: If solver name is invalid
        """
        if solver_name not in self.SOLVERS:
            raise ValueError(f"Unknown solver: {solver_name}")

        return self.SOLVERS[solver_name].exec_path

    def is_solver_available(self, solver_name: str) -> bool:
        """
        Check if a solver is available.

        Args:
            solver_name: Name of the solver

        Returns:
            True if solver is available, False otherwise

        Raises:
            ValueError: If solver name is invalid
        """
        if solver_name not in self.SOLVERS:
            raise ValueError(f"Unknown solver: {solver_name}")

        return self.SOLVERS[solver_name].is_available


# Global singleton instance
global_config = GlobalConfig()
# print(global_config.get_solver_path("z3"))
