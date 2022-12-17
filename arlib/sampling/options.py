import abc


class SamplerOptions(object):
    """
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, generate_models=True, incremental=True, random_seed=None,
                 sampler_options=None):

        if generate_models not in (True, False):
            raise ValueError("Invalid value %s for 'generate_models'" \
                             % generate_models)
        self.generate_models = generate_models

        if incremental not in (True, False):
            raise ValueError("Invalid value %s for 'incremental'" \
                             % incremental)
        self.incremental = incremental

        if random_seed is not None and type(random_seed) != int:
            raise ValueError("Invalid value %s for 'random_seed'" \
                             % random_seed)
        self.random_seed = random_seed

        if sampler_options is not None:
            try:
                sampler_options = dict(sampler_options)
            except:
                raise ValueError("Invalid value %s for 'sampler_options'" \
                                 % sampler_options)
        else:
            sampler_options = dict()
        self.sampler_options = sampler_options

    @abc.abstractmethod
    def __call__(self, sampler):
        """Handle the setting options within sampler"""
        raise NotImplementedError

    def as_kwargs(self):
        """Construct a dictionary object that can be used as **kwargs.
        This can be used to duplicate the options.
        """
        kwargs = {}
        for k in ("generate_models", "incremental", "unsat_cores_mode",
                  "random_seed", "sampler_options"):
            v = getattr(self, k)
            kwargs[k] = v
        return kwargs

# EOC samplerOptions
