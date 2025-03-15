"""Abstract Interfaces for Samplera"""

from enum import Enum
from typing import List, Dict, Any, Optional
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
                 stats: Optional[Dict[str, Any]] = None):
        self.samples = samples
        self.stats = stats or {}
        self.success = len(samples) > 0


class SamplingOptions:
    def __init__(self,
                 method: SamplingMethod = SamplingMethod.ENUMERATION,
                 num_samples: int = 1,
                 timeout: float = None,
                 random_seed: int = None,
                 **kwargs):
        self.method = method
        self.num_samples = num_samples
        self.timeout = timeout
        self.random_seed = random_seed
        self.additional_options = kwargs


class Sampler(ABC):
    """Abstract base class for all samplers."""
    
    @abstractmethod
    def supports_logic(self, logic: Logic) -> bool:
        """Check if this sampler supports the given logic."""
        pass

    @abstractmethod
    def init_from_formula(self, formula: z3.ExprRef):
        """Initialize sampler with a formula."""
        pass

    @abstractmethod
    def sample(self, options: SamplingOptions) -> SamplingResult:
        """Generate samples according to the given options."""
        pass

    def get_supported_methods(self) -> List[SamplingMethod]:
        """Return list of supported sampling methods."""
        pass

