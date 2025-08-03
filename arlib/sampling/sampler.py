"""Abstract Interfaces for Samplera"""

from enum import Enum
from typing import List, Dict, Any, Optional, Union, Set
from abc import ABC, abstractmethod
import z3


class Logic(Enum):
    QF_BOOL = "QF_BOOL"
    QF_BV = "QF_BV"
    QF_LRA = "QF_LRA"
    QF_LIA = "QF_LIA"
    QF_NRA = "QF_NRA"
    QF_NIA = "QF_NIA"
    QF_LIRA = "QF_LIRA"
    QF_ALL = "QF_ALL"


class SamplingMethod(Enum):
    ENUMERATION = "enumeration"
    MCMC = "mcmc"
    REGION = "region"
    SEARCH_TREE = "search_tree"
    HASH_BASED = "hash_based"
    DIKIN_WALK = "dikin_walk"


class SamplingResult:
    def __init__(self, samples: List[Dict[str, Any]],
                 stats: Optional[Dict[str, Any]] = None) -> None:
        self.samples: List[Dict[str, Any]] = samples
        self.stats: Dict[str, Any] = stats or {}
        self.success: bool = len(samples) > 0

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a specific sample by index."""
        return self.samples[index]

    def __iter__(self):
        """Iterate over the samples."""
        return iter(self.samples)

    def __str__(self) -> str:
        """String representation of the sampling result."""
        if not self.success:
            return "SamplingResult(success=False, samples=[])"

        sample_str = f"{len(self.samples)} samples"
        if len(self.samples) > 0:
            sample_str += f", first sample: {self.samples[0]}"

        return f"SamplingResult(success={self.success}, {sample_str})"


class SamplingOptions:
    def __init__(self,
                 method: SamplingMethod = SamplingMethod.ENUMERATION,
                 num_samples: int = 1,
                 timeout: Optional[float] = None,
                 random_seed: Optional[int] = None,
                 **kwargs: Any) -> None:
        self.method: SamplingMethod = method
        self.num_samples: int = num_samples
        self.timeout: Optional[float] = timeout
        self.random_seed: Optional[int] = random_seed
        self.additional_options: Dict[str, Any] = kwargs


class Sampler(ABC):
    """Abstract base class for all samplers."""

    @abstractmethod
    def supports_logic(self, logic: Logic) -> bool:
        """Check if this sampler supports the given logic."""
        pass

    @abstractmethod
    def init_from_formula(self, formula: z3.ExprRef) -> None:
        """Initialize sampler with a formula."""
        pass

    @abstractmethod
    def sample(self, options: SamplingOptions) -> SamplingResult:
        """Generate samples according to the given options."""
        pass

    def get_supported_methods(self) -> Set[SamplingMethod]:
        """Return set of supported sampling methods."""
        return set()

    def get_supported_logics(self) -> Set[Logic]:
        """Return set of supported logics."""
        return set()
