"""Main class definition for the ReducedProduct conjunctive domain.
"""
from typing import Any, List, Dict
import z3
from ..algorithms import bilateral
from ..z3_variables import Z3VariablesDomain
from .abstract import ReducedProductAbstractState


# TODO(masotoud): find a different naming scheme that makes this clearer, if
# possible.
# pylint: disable=invalid-name


class ReducedProductDomain(Z3VariablesDomain):
    """Represents an abstract space combining information from two others.

    For example, you may track both the interval and parity of a set of integer
    variables and want to use information from one analysis to improve the
    information in another.
    """

    def __init__(self, variables: List[str], domain_A: Any, domain_B: Any) -> None:
        """Construct a ReducedProductDomain with given variables, sub-domains.

        @domain_A, @domain_B should be instantiated Z3VariablesDomains with the
        same variables as @variables.
        """
        Z3VariablesDomain.__init__(self, variables, z3.Int)
        self.domain_A = domain_A
        self.domain_B = domain_B

    def gamma_hat(self, alpha: ReducedProductAbstractState) -> Any:
        """Returns a formula describing the same states as alpha.
        """
        return z3.And(self.domain_A.gamma_hat(alpha.state_A),
                      self.domain_B.gamma_hat(alpha.state_B))

    def join(self, elements: List[ReducedProductAbstractState]) -> ReducedProductAbstractState:
        """Returns the join of a set of abstract states.

        join([ alpha_1, alpha_2, ..., alpha_n ]) is the smallest alpha
        containing all alpha_1, ..., alpha_n. It may not be in elements.

        This method DOES reduce after joining.
        """
        elements_A = [element.state_A for element in elements]
        elements_B = [element.state_B for element in elements]
        joined_A = self.domain_A.join(elements_A)
        joined_B = self.domain_B.join(elements_B)
        joined = ReducedProductAbstractState(joined_A, joined_B)
        return self.reduce(joined)

    def meet(self, elements: List[ReducedProductAbstractState]) -> ReducedProductAbstractState:
        """Returns the meet of a set of abstract states.

        join([ alpha_1, alpha_2, ..., alpha_n ]) is the greatest alpha
        contained by all alpha_1, ..., alpha_n. It may not be in elements.

        This method DOES NOT reduce after meeting.

        TODO(masotoud): We do not reduce here because it can cause issues when
        used inside of bilateral (essentially, we'll often meet with Top or
        something close to Top, so we end up with infinite ascending chains
        when we try to reduce). In the future we should clarify this, perhaps
        having separate meet() and meet_reduce() operations, and making sure
        join() has similar behavior.
        """
        elements_A = [element.state_A for element in elements]
        elements_B = [element.state_B for element in elements]
        met_A = self.domain_A.meet(elements_A)
        met_B = self.domain_B.meet(elements_B)
        met = ReducedProductAbstractState(met_A, met_B)
        return met

    def abstract_consequence(self, lower: ReducedProductAbstractState, upper: ReducedProductAbstractState) -> ReducedProductAbstractState:
        """Returns the "abstract consequence" of lower and upper.

        The abstract consequence must be a superset of lower and *NOT* a
        superset of upper.

        TODO(masotoud): ensure this is correct.
        """
        consequence_A = self.domain_A.abstract_consequence(
            lower.state_A, upper.state_A)
        consequence_B = self.domain_B.abstract_consequence(
            lower.state_B, upper.state_B)
        return ReducedProductAbstractState(consequence_A, consequence_B)

    def beta(self, sigma: Any) -> ReducedProductAbstractState:
        """Returns the least abstract state describing sigma.

        Sigma should be an Z3VariablesState. See Definition 3.4 in:
        Thakur, A. V. (2014, August). Symbolic Abstraction: Algorithms and
        Applications (Ph.D. dissertation). Computer Sciences Department,
        University of Wisconsin, Madison.
        """
        beta_A = self.domain_A.beta(sigma)
        beta_B = self.domain_B.beta(sigma)
        return ReducedProductAbstractState(beta_A, beta_B)

    @property
    def top(self) -> ReducedProductAbstractState:
        return ReducedProductAbstractState(self.domain_A.top, self.domain_B.top)

    @property
    def bottom(self) -> ReducedProductAbstractState:
        return ReducedProductAbstractState(self.domain_A.bottom, self.domain_B.bottom)

    def reduce(self, alpha: ReducedProductAbstractState) -> ReducedProductAbstractState:
        return alpha

    def translate(self, translation: Dict[str, str]) -> 'ReducedProductDomain':
        return ReducedProductDomain(
            self.variables,
            self.domain_A.translate(translation),
            self.domain_B.translate(translation)
        )
