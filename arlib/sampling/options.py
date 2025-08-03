import abc
from typing import Dict, Any, Optional, Union


class SamplerOptions(object):
    """
    Base class for sampler options configuration.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, generate_models: bool = True, incremental: bool = True,
                 random_seed: Optional[int] = None,
                 sampler_options: Optional[Dict[str, Any]] = None) -> None:

        if generate_models not in (True, False):
            raise ValueError(f"Invalid value {generate_models} for 'generate_models'")
        self.generate_models: bool = generate_models

        if incremental not in (True, False):
            raise ValueError(f"Invalid value {incremental} for 'incremental'")
        self.incremental: bool = incremental

        if random_seed is not None and not isinstance(random_seed, int):
            raise ValueError(f"Invalid value {random_seed} for 'random_seed'")
        self.random_seed: Optional[int] = random_seed

        if sampler_options is not None:
            try:
                sampler_options = dict(sampler_options)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid value {sampler_options} for 'sampler_options'")
        else:
            sampler_options = dict()
        self.sampler_options: Dict[str, Any] = sampler_options

    @abc.abstractmethod
    def __call__(self, sampler: Any) -> None:
        """Handle the setting options within sampler"""
        raise NotImplementedError

    def as_kwargs(self) -> Dict[str, Any]:
        """Construct a dictionary object that can be used as **kwargs.
        This can be used to duplicate the options.
        """
        kwargs: Dict[str, Any] = {}
        for k in ("generate_models", "incremental", "unsat_cores_mode",
                  "random_seed", "sampler_options"):
            v = getattr(self, k, None)
            kwargs[k] = v
        return kwargs

# EOC samplerOptions
