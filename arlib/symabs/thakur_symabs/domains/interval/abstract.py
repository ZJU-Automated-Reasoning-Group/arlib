"""Definitions for the Intervals abstract space.
"""
from ..core.abstract import AbstractState


class Interval:
    """Represents a single integer, closed interval.
    """

    def __init__(self, lower, upper):
        """Construct an Interval with the given lower and upper bounds.

        We take lower and upper to be both INCLUSIVE.
        """
        self.lower = lower
        self.upper = upper

    def __le__(self, rhs):
        """True iff @self represents a subset of the integers of @rhs.
        """
        return (self.lower > self.upper or
                rhs.lower <= self.lower <= self.upper <= rhs.upper)

    def __ge__(self, rhs):
        """True iff @rhs represents a superset of the integers as @self.
        """
        return rhs <= self

    def __eq__(self, rhs):
        """True iff this interval represents the same integers as @rhs.
        """
        return self <= rhs <= self

    def union(self, other):
        """Returns the smallest Interval containing both self and other.
        """
        return Interval(min(self.lower, other.lower),
                        max(self.upper, other.upper))

    def intersection(self, other):
        """Returns the largest Interval contained in both self and other.
        """
        return Interval(max(self.lower, other.lower),
                        min(self.upper, other.upper))

    def __repr__(self):
        """Human-readable representation of the interval.
        """
        return str((self.lower, self.upper))


class IntervalAbstractState(AbstractState):
    """Abstract state describing the intervals of a collection of variables.
    """

    def __init__(self, variable_intervals):
        """Initializer for the IntervalAbstractState.

        @variable_intervals should be {"variable_name": (lower, upper)}, where
        @lower, @upper are integers. float("+inf") and float("-inf") should be
        used for +/- infinity.
        """
        self.variable_intervals = variable_intervals
        self.variables = list(variable_intervals.keys())

    def copy(self):
        """Returns a new IntervalAbstractState with the same intervals as self.
        """
        return IntervalAbstractState(self.variable_intervals.copy())

    def interval_of(self, variable_name):
        """Returns the Interval of a variable given its name.
        """
        return self.variable_intervals[variable_name]

    def set_interval(self, variable_name, interval):
        """Sets the interval of a variable given its name.
        """
        self.variable_intervals[variable_name] = interval

    def __le__(self, rhs):
        """True if self represents a subset of rhs.

        Note that this definition means (not (a <= b)) does NOT imply a > b.
        Perhaps we should raise an exception when elements are uncomparable.
        This assumes that both have the exact same set of variables, and does
        not check that condition.
        """
        return all(self.interval_of(name) <= rhs.interval_of(name)
                   for name in self.variables)

    def translate(self, translation):
        """Translate the state to use different variable names.
        """
        return IntervalAbstractState({
            translation[name]: state
            for name, state in self.variable_intervals.items()
        })

    def __str__(self):
        """Returns a human-readable representation of the state.
        """
        return str(self.variable_intervals)
