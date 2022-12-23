"""Main class definition for the ReducedProduct conjunctive domain.
"""
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

    def __init__(self, variables, domain_A, domain_B):
        """Construct a ReducedProductDomain with given variables, sub-domains.

        @domain_A, @domain_B should be instantiated Z3VariablesDomains with the
        same variables as @variables.
        """
        Z3VariablesDomain.__init__(self, variables, z3.Int)
        self.domain_A = domain_A
        self.domain_B = domain_B

    def gamma_hat(self, alpha):
        """Returns a formula describing the same states as alpha.
        """
        return z3.And(self.domain_A.gamma_hat(alpha.state_A),
                      self.domain_B.gamma_hat(alpha.state_B))

    def join(self, elements):
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

    def meet(self, elements):
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

    def abstract_consequence(self, lower, upper):
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

    # Converts one concrete set of variables into an abstract element
    def beta(self, sigma):
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
    def top(self):
        """Returns the least upper bound of the entire abstract space.
        """
        top_A = self.domain_A.top
        top_B = self.domain_B.top
        return ReducedProductAbstractState(top_A, top_B)

    @property
    def bottom(self):
        """Returns the greatest lower bound of the entire abstract space.
        """
        bottom_A = self.domain_A.bottom
        bottom_B = self.domain_B.bottom
        return ReducedProductAbstractState(bottom_A, bottom_B)

    def reduce(self, alpha):
        """'Tightens' the consitutient states as much as possible.

        For example, given ReducedProduct<Sign, Interval>, calling
        Reduce(Positive, [-5, 5]) should give (Positive, [1, 5]).
        """
        reduced_A = bilateral(self.domain_A, self.gamma_hat(alpha))
        reduced_B = bilateral(self.domain_B, self.gamma_hat(alpha))
        return ReducedProductAbstractState(reduced_A, reduced_B)

    def translate(self, translation):
        """Returns a new domain with the variable names translated.
        """
        variables = list(map(translation.__getitem__, self.variables))
        domain_A = self.domain_A.translate(translation)
        domain_B = self.domain_B.translate(translation)
        return ReducedProductDomain(variables, domain_A, domain_B)
