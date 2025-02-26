"""
Sampler for various logical constraints
"""

from arlib.sampling.finite_domain.bool_sampler import BooleanSampler
from arlib.sampling.finite_domain.bv_sampler import BitVectorSampler
from arlib.sampling.linear_ira.lira_sampler import LIRASampler
from arlib.sampling.utils.sampler import Logic, Sampler


class SamplerFactory:
    """Factory class to create appropriate sampler instances."""
    
    @staticmethod
    def create_sampler(logic: Logic, 
                      method: SamplingMethod = SamplingMethod.ENUMERATION) -> Sampler:
        """Create a sampler instance based on logic and method."""

        samplers = {
            Logic.QF_BOOL: BooleanSampler,
            Logic.QF_BV: BitVectorSampler,
            Logic.QF_LRA: LIRASampler,
            Logic.QF_LIA: LIRASampler,
            Logic.QF_LIRA: LIRASampler,
        }
        
        sampler_cls = samplers.get(logic)
        if not sampler_cls:
            raise ValueError(f"No sampler available for logic {logic}")
            
        return sampler_cls()


def sample_formula(formula: z3.ExprRef,
                  logic: Logic,
                  options: SamplingOptions = None) -> SamplingResult:
    """High-level API for sampling from a formula."""
    if options is None:
        options = SamplingOptions()
        
    sampler = SamplerFactory.create_sampler(logic, options.method)
    sampler.init_from_formula(formula)
    return sampler.sample(options)
    

def demo_sampler():
    x, y = z3.Reals("x y")
    fml = z3.And(x + y > 0, x - y < 1)
    sampler = SamplerFactory.create_sampler(Logic.QF_LRA)
    sampler.init_from_formula(fml)
    options = SamplingOptions(num_samples=10)
    res = sampler.sample(options)
    print(res.samples)

if __name__ == "__main__":
    demo_sampler()
