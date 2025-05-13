"""
Symbolic Finite Automata (SFA) implementation using Z3 solver.

Example:
    >>> # Create a simple SFA that accepts strings starting with 'a'
    >>> from z3 import String, StringVal, Length, SubString
    >>> states = {'q0', 'q1'}
    >>> x = String('x')  # Create a variable for the predicate
    >>> predicates = {'starts_a': SubString(x, 0, 1) == StringVal('a')}  # Check first character
    >>> transitions = {('q0', 'starts_a'): 'q1'}
    >>> sfa = SFA(states, predicates, transitions, 'q0', {'q1'}, x.sort())
    >>> sfa.accepts([StringVal('abc')])
    True
    >>> sfa.accepts([StringVal('bcd')])
    False
"""
from typing import Set, Dict, Tuple, Any, List, Callable, TypeVar
from collections import defaultdict
from z3 import *

T = TypeVar('T')  # Type of the alphabet elements


class SFA:
    def __init__(self, states: Set[str],
                 predicates: Dict[str, ExprRef],  # Changed to Z3 expressions
                 transitions: Dict[Tuple[str, str], str],
                 initial_state: str,
                 accepting_states: Set[str],
                 sort: SortRef):  # Added sort parameter for the alphabet domain
        """Initialize SFA with Z3 predicates"""
        # Validate states
        if not states:
            raise ValueError("States set cannot be empty")

        if initial_state not in states:
            raise ValueError(f"Initial state '{initial_state}' not in states")

        if not accepting_states.issubset(states):
            invalid = accepting_states - states
            raise ValueError(f"Accepting states {invalid} not in states")

        # Validate predicates are boolean expressions
        for name, pred in predicates.items():
            if not is_bool(pred):
                raise ValueError(f"Predicate '{name}' must be a boolean expression")

        # Validate transitions
        for (state, pred), next_state in transitions.items():
            if state not in states:
                raise ValueError(f"Transition from invalid state '{state}'")
            if pred not in predicates:
                raise ValueError(f"Unknown predicate '{pred}'")
            if next_state not in states:
                raise ValueError(f"Transition to invalid state '{next_state}'")

        self.states = states
        self.predicates = predicates
        self.transitions = transitions
        self.initial_state = initial_state
        self.accepting_states = accepting_states
        self.sort = sort
        self._solver = Solver()

    def accepts(self, word: List[ExprRef]) -> bool:
        """
        Check if the SFA accepts a word of Z3 expressions.
        
        Args:
            word: List of Z3 expressions representing the input word
            
        Returns:
            bool: True if the word is accepted, False otherwise
            
        Example:
            >>> from z3 import Int, IntVal
            >>> states = {'q0', 'q1'}
            >>> x = Int('x')  # Create a variable for the predicate
            >>> predicates = {'is_positive': x > 0}
            >>> transitions = {('q0', 'is_positive'): 'q1'}
            >>> sfa = SFA(states, predicates, transitions, 'q0', {'q1'}, x.sort())
            >>> sfa.accepts([IntVal(5)])
            True
            >>> sfa.accepts([IntVal(-1)])
            False
        """
        if not word:
            return self.initial_state in self.accepting_states

        current = self.initial_state
        x = Const('x', self.sort)  # Create variable once

        for symbol in word:
            if symbol.sort() != self.sort:
                return False

            moved = False
            for (state, pred), next_state in self.transitions.items():
                if state == current:
                    self._solver.push()
                    subst_pred = substitute(self.predicates[pred], (x, symbol))
                    self._solver.add(subst_pred)
                    if self._solver.check() == sat:
                        current = next_state
                        moved = True
                        self._solver.pop()
                        break
                    self._solver.pop()

            if not moved:
                return False

        return current in self.accepting_states

    def is_empty(self) -> bool:
        """Check if the language accepted by the SFA is empty"""
        visited = set()
        work_list = [self.initial_state]

        while work_list:
            current = work_list.pop()
            if current in self.accepting_states:
                return False

            if current in visited:
                continue
            visited.add(current)

            for (state, _), next_state in self.transitions.items():
                if state == current and next_state not in visited:
                    work_list.append(next_state)

        return True

    def is_universal(self, domain_constraint: ExprRef = None) -> bool:
        """
        Check if the SFA accepts all words over the domain
        
        Args:
            domain_constraint: Optional Z3 expression defining the domain
        """
        # Create complement automaton
        complement = self.complement(domain_constraint)
        return complement.is_empty()

    def complement(self, domain_constraint: ExprRef = None) -> 'SFA':
        """
        Construct complement SFA
        
        Args:
            domain_constraint: Optional Z3 expression defining the domain
        """
        # Add sink state for completeness
        new_states = self.states | {'sink'}

        # Add domain constraint predicate if provided
        new_predicates = self.predicates.copy()
        if domain_constraint is not None:
            new_predicates['domain'] = domain_constraint

        # Initialize new transitions dictionary with copy of existing transitions
        new_transitions = self.transitions.copy()

        # Create negated predicates for uncovered transitions
        x = Const('x', self.sort)
        for state in self.states:
            # Collect all predicates for transitions from this state
            state_predicates = []
            for (s, p) in self.transitions:
                if s == state:
                    state_predicates.append(self.predicates[p])

            # Create a predicate representing all outgoing transitions
            if state_predicates:
                covered_expr = state_predicates[0]
                for pred in state_predicates[1:]:
                    covered_expr = Or(covered_expr, pred)

                # Add negation of covered predicates
                pred_name = f'not_covered_{state}'
                new_predicates[pred_name] = Not(covered_expr)
                new_transitions[(state, pred_name)] = 'sink'
            else:
                # If no transitions, add a true predicate to sink
                pred_name = f'always_{state}'
                new_predicates[pred_name] = BoolVal(True)
                new_transitions[(state, pred_name)] = 'sink'

        # Add self-loops on sink state
        for p in new_predicates:
            new_transitions[('sink', p)] = 'sink'

        # Compute new accepting states:
        # 1. Exclude original accepting states
        # 2. Include the sink state (to accept inputs rejected by original SFA)
        new_accepting = (new_states - self.accepting_states)

        return SFA(
            new_states,
            new_predicates,
            new_transitions,
            self.initial_state,
            new_accepting,
            self.sort
        )

    def intersect(self, other: 'SFA') -> 'SFA':
        """Construct product SFA accepting intersection of languages"""
        new_states = {f"{s1},{s2}"
                      for s1 in self.states
                      for s2 in other.states}

        new_predicates = {}
        for p1, f1 in self.predicates.items():
            for p2, f2 in other.predicates.items():
                name = f"{p1}&{p2}"
                new_predicates[name] = lambda x, f1=f1, f2=f2: f1(x) and f2(x)

        new_transitions = {}
        for (s1, p1), n1 in self.transitions.items():
            for (s2, p2), n2 in other.transitions.items():
                new_transitions[(f"{s1},{s2}", f"{p1}&{p2}")] = f"{n1},{n2}"

        new_initial = f"{self.initial_state},{other.initial_state}"
        new_accepting = {f"{s1},{s2}"
                         for s1 in self.accepting_states
                         for s2 in other.accepting_states}

        # Fix: Pass sort to constructor
        return SFA(new_states, new_predicates, new_transitions,
                   new_initial, new_accepting, self.sort)

    def minimize(self) -> 'SFA':
        """Minimize SFA using predicate-based Hopcroft's algorithm"""
        # Initial partition: accepting and non-accepting states
        partition = [self.accepting_states, self.states - self.accepting_states]
        partition = [s for s in partition if s]

        while True:
            new_partition = []
            for group in partition:
                splits = defaultdict(set)
                for state in group:
                    # Create signature based on transitions
                    sig = []
                    for pred in sorted(self.predicates):
                        next_states = set()
                        for (s, p), n in self.transitions.items():
                            if s == state and p == pred:
                                next_states.add(n)
                        # Find which group contains the next states
                        group_nums = []
                        for ns in next_states:
                            for i, g in enumerate(partition):
                                if ns in g:
                                    group_nums.append(i)
                        sig.append(Tuple(sorted(group_nums)))
                    splits[Tuple(sig)].add(state)
                new_partition.extend(splits.values())

            if len(new_partition) == len(partition):
                break
            partition = new_partition

        # Build minimized SFA
        state_map = {}
        for i, group in enumerate(partition):
            for state in group:
                state_map[state] = f"q{i}"

        new_states = set(state_map.values())
        new_transitions = {}
        for (state, pred), next_state in self.transitions.items():
            new_transitions[(state_map[state], pred)] = state_map[next_state]

        new_accepting = {state_map[s] for s in self.accepting_states}
        new_initial = state_map[self.initial_state]

        return SFA(new_states, self.predicates, new_transitions,
                   new_initial, new_accepting, self.sort)

    def add_transition(self, from_state: str, predicate: str, to_state: str) -> None:
        """
        Add a new transition to the SFA.
        
        Args:
            from_state: Source state
            predicate: Predicate name
            to_state: Target state
        """
        if from_state not in self.states:
            raise ValueError(f"Invalid source state '{from_state}'")
        if predicate not in self.predicates:
            raise ValueError(f"Unknown predicate '{predicate}'")
        if to_state not in self.states:
            raise ValueError(f"Invalid target state '{to_state}'")

        self.transitions[(from_state, predicate)] = to_state


# Add tests at the end of the file
def run_tests():
    """Run basic tests for the SFA implementation."""
    from z3 import Int, IntVal, String, StringVal, SubString

    # Test 1: Integer SFA accepting positive numbers
    states = {'q0', 'q1'}
    x = Int('x')  # Create a variable for the predicate
    predicates = {'is_positive': x > 0}
    transitions = {('q0', 'is_positive'): 'q1'}
    sfa = SFA(states, predicates, transitions, 'q0', {'q1'}, x.sort())

    assert sfa.accepts([IntVal(5)])
    assert not sfa.accepts([IntVal(-1)])
    assert not sfa.accepts([IntVal(0)])

    # Test 2: String SFA accepting strings starting with 'a'
    states = {'q0', 'q1'}
    x = String('x')  # Create a variable for the predicate
    predicates = {'starts_a': SubString(x, 0, 1) == StringVal('a')}  # Check first character
    transitions = {('q0', 'starts_a'): 'q1'}
    sfa = SFA(states, predicates, transitions, 'q0', {'q1'}, x.sort())

    assert sfa.accepts([StringVal('abc')])
    assert not sfa.accepts([StringVal('bcd')])

    # Test 3: Test empty word handling
    assert not sfa.accepts([])  # q0 is not accepting

    # Test 4: Test complement with more specific cases
    states = {'q0', 'q1'}
    x = String('x')
    predicates = {'starts_a': SubString(x, 0, 1) == StringVal('a')}
    transitions = {('q0', 'starts_a'): 'q1'}
    sfa = SFA(states, predicates, transitions, 'q0', {'q1'}, x.sort())

    complement = sfa.complement()
    # String starting with 'a' should not be accepted by complement
    assert not complement.accepts([StringVal('abc')])
    # String starting with 'b' should be accepted by complement
    assert complement.accepts([StringVal('bcd')])
    # Empty string should be accepted by complement (since original rejects it)
    assert complement.accepts([])

    print("All tests passed!")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    run_tests()
