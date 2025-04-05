"""
Provide external interface for computing unsat cores that encapsulates
marco.py, musx.py, optux.py (and other possible new implementations)

NOTICE: distinguish the following types of problems:
    (1) unsat core: a list of literals that is unsatisfiable
    (2) minimal unsatisfiable subset (MUS): a subset of literals that is unsatisfiable
                                           and minimal (removing any constraint makes it satisfiable)
    (3) MUS enumeration: enumerating all MUSs
"""

from enum import Enum
import importlib.util
import os
import sys
from typing import List, Set, Dict, Any, Optional, Tuple, Callable, Union


class Algorithm(Enum):
    """Enumeration of available unsat core algorithms."""
    MARCO = "marco"
    MUSX = "musx"
    OPTUX = "optux"

    @classmethod
    def from_string(cls, name: str) -> "Algorithm":
        """Convert string to Algorithm enum."""
        name = name.lower()
        for alg in cls:
            if alg.value == name:
                return alg
        raise ValueError(f"Unknown algorithm: {name}")


class UnsatCoreResult:
    """Result of an unsat core computation."""

    def __init__(self,
                 cores: List[Set[int]],
                 is_minimal: bool = False,
                 stats: Optional[Dict[str, Any]] = None):
        """
        Initialize UnsatCoreResult.
        
        Args:
            cores: List of unsat cores, each represented as a set of constraint indices
            is_minimal: Whether the cores are guaranteed to be minimal
            stats: Optional statistics about the computation
        """
        self.cores = cores
        self.is_minimal = is_minimal
        self.stats = stats or {}

    def __str__(self) -> str:
        """String representation of the result."""
        cores_str = "\n".join([f"Core {i + 1}: {sorted(core)}" for i, core in enumerate(self.cores)])
        minimal_str = "minimal" if self.is_minimal else "not necessarily minimal"
        return f"Found {len(self.cores)} {minimal_str} unsat cores:\n{cores_str}"


class UnsatCoreComputer:
    """Interface for computing unsat cores."""

    def __init__(self, algorithm: Union[str, Algorithm] = Algorithm.MARCO):
        """
        Initialize UnsatCoreComputer.
        
        Args:
            algorithm: Algorithm to use for computing unsat cores
        """
        if isinstance(algorithm, str):
            self.algorithm = Algorithm.from_string(algorithm)
        else:
            self.algorithm = algorithm

        # Dynamically import the appropriate module
        self._load_algorithm_module()

    def _load_algorithm_module(self):
        """Load the module for the selected algorithm."""
        module_name = self.algorithm.value
        try:
            # Try to import from the same package
            module_path = os.path.join(os.path.dirname(__file__), f"{module_name}.py")
            if os.path.exists(module_path):
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec and spec.loader:
                    self.module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(self.module)
                else:
                    raise ImportError(f"Failed to load {module_name} module")
            else:
                # Fall back to regular import
                self.module = importlib.import_module(f"arlib.unsat_core.{module_name}")
        except ImportError as e:
            raise ImportError(f"Failed to import algorithm module {module_name}: {e}")

    def compute_unsat_core(self,
                           constraints: List[Any],
                           solver_factory: Callable[[], Any],
                           timeout: Optional[int] = None,
                           **kwargs) -> UnsatCoreResult:
        """
        Compute an unsat core for the given constraints.
        
        Args:
            constraints: List of constraints to find unsat core from
            solver_factory: Function that creates a new solver instance
            timeout: Optional timeout in seconds
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            UnsatCoreResult containing the computed unsat core(s)
        """
        if self.algorithm == Algorithm.MARCO:
            return self._run_marco(constraints, solver_factory, timeout, **kwargs)
        elif self.algorithm == Algorithm.MUSX:
            return self._run_musx(constraints, solver_factory, timeout, **kwargs)
        elif self.algorithm == Algorithm.OPTUX:
            return self._run_optux(constraints, solver_factory, timeout, **kwargs)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    def _run_marco(self,
                   constraints: List[Any],
                   solver_factory: Callable[[], Any],
                   timeout: Optional[int] = None,
                   max_cores: int = 1,
                   **kwargs) -> UnsatCoreResult:
        """Run the MARCO algorithm."""
        cores = self.module.find_unsat_cores(
            constraints=constraints,
            solver_factory=solver_factory,
            timeout=timeout,
            max_cores=max_cores,
            **kwargs
        )
        return UnsatCoreResult(cores=cores, is_minimal=True)

    def _run_musx(self,
                  constraints: List[Any],
                  solver_factory: Callable[[], Any],
                  timeout: Optional[int] = None,
                  **kwargs) -> UnsatCoreResult:
        """Run the MUSX algorithm."""
        core = self.module.compute_minimal_unsat_core(
            constraints=constraints,
            solver_factory=solver_factory,
            timeout=timeout,
            **kwargs
        )
        return UnsatCoreResult(cores=[core], is_minimal=True)

    def _run_optux(self,
                   constraints: List[Any],
                   solver_factory: Callable[[], Any],
                   timeout: Optional[int] = None,
                   **kwargs) -> UnsatCoreResult:
        """Run the OPTUX algorithm."""
        core, stats = self.module.compute_minimum_unsat_core(
            constraints=constraints,
            solver_factory=solver_factory,
            timeout=timeout,
            **kwargs
        )
        return UnsatCoreResult(cores=[core], is_minimal=True, stats=stats)

    def enumerate_all_mus(self,
                          constraints: List[Any],
                          solver_factory: Callable[[], Any],
                          timeout: Optional[int] = None,
                          **kwargs) -> UnsatCoreResult:
        """
        Enumerate all Minimal Unsatisfiable Subsets (MUSes).
        
        Args:
            constraints: List of constraints
            solver_factory: Function that creates a new solver instance
            timeout: Optional timeout in seconds
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            UnsatCoreResult containing all MUSes
        """
        if self.algorithm != Algorithm.MARCO:
            # MARCO is the only algorithm that supports MUS enumeration
            self.algorithm = Algorithm.MARCO
            self._load_algorithm_module()

        cores = self.module.find_unsat_cores(
            constraints=constraints,
            solver_factory=solver_factory,
            timeout=timeout,
            enumerate_all=True,
            **kwargs
        )
        return UnsatCoreResult(cores=cores, is_minimal=True)


def get_unsat_core(constraints: List[Any],
                   solver_factory: Callable[[], Any],
                   algorithm: Union[str, Algorithm] = "marco",
                   timeout: Optional[int] = None,
                   **kwargs) -> UnsatCoreResult:
    """
    Convenience function to compute an unsat core.
    
    Args:
        constraints: List of constraints
        solver_factory: Function that creates a new solver instance
        algorithm: Algorithm to use (default: "marco")
        timeout: Optional timeout in seconds
        **kwargs: Additional algorithm-specific parameters
        
    Returns:
        UnsatCoreResult containing the computed unsat core(s)
    """
    computer = UnsatCoreComputer(algorithm)
    return computer.compute_unsat_core(constraints, solver_factory, timeout, **kwargs)


def enumerate_all_mus(constraints: List[Any],
                      solver_factory: Callable[[], Any],
                      timeout: Optional[int] = None,
                      **kwargs) -> UnsatCoreResult:
    """
    Convenience function to enumerate all MUSes.
    
    Args:
        constraints: List of constraints
        solver_factory: Function that creates a new solver instance
        timeout: Optional timeout in seconds
        **kwargs: Additional algorithm-specific parameters
        
    Returns:
        UnsatCoreResult containing all MUSes
    """
    computer = UnsatCoreComputer(Algorithm.MARCO)
    return computer.enumerate_all_mus(constraints, solver_factory, timeout, **kwargs)
