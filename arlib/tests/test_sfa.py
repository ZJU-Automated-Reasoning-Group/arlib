import unittest
from typing import List, Set, Dict, Tuple
from z3 import *
from arlib.automata.sfa import SFA


class TestSFA(unittest.TestCase):
    def setUp(self):
        # Create a simple SFA that accepts even integers
        self.states = {'q0', 'q1'}
        self.sort = IntSort()
        x = Int('x')
        
        self.predicates = {
            'even': x % 2 == 0,
            'odd': x % 2 == 1
        }
        
        self.transitions = {
            ('q0', 'even'): 'q1',
            ('q0', 'odd'): 'q0',
            ('q1', 'even'): 'q1',
            ('q1', 'odd'): 'q0'
        }
        
        self.initial_state = 'q0'
        self.accepting_states = {'q1'}
        
        self.sfa = SFA(
            self.states,
            self.predicates,
            self.transitions,
            self.initial_state,
            self.accepting_states,
            self.sort
        )

    def test_accepts_valid_even_sequence(self):
        """Test accepting sequence of even integers"""
        word = [IntVal(2), IntVal(4), IntVal(6)]
        self.assertTrue(self.sfa.accepts(word))

    def test_rejects_odd_sequence(self):
        """Test rejecting sequence ending with odd integer"""
        word = [IntVal(2), IntVal(4), IntVal(3)]
        self.assertFalse(self.sfa.accepts(word))

    def test_accepts_empty_word(self):
        """Test empty word (should be rejected as initial state is not accepting)"""
        self.assertFalse(self.sfa.accepts([]))

    def test_rejects_invalid_sort(self):
        """Test rejection of symbols with wrong sort"""
        word = [RealVal(2.0)]  # Real instead of Int
        self.assertFalse(self.sfa.accepts(word))

    def test_accepts_mixed_even_sequence(self):
        """Test accepting sequence with mixed even numbers"""
        word = [IntVal(1), IntVal(2), IntVal(4)]
        self.assertTrue(self.sfa.accepts(word))

    def test_rejects_unsatisfiable_predicate(self):
        """Test rejection when no predicate is satisfied"""
        # Create SFA with unsatisfiable predicate
        x = Int('x')
        predicates = {'impossible': And(x > 0, x < 0)}
        
        sfa = SFA(
            {'q0'},
            predicates,
            {('q0', 'impossible'): 'q0'},
            'q0',
            {'q0'},
            IntSort()
        )
        
        word = [IntVal(0)]
        self.assertFalse(sfa.accepts(word))

    def test_multiple_possible_transitions(self):
        """Test behavior with multiple possible transitions"""
        # Create SFA with overlapping predicates
        x = Int('x')
        predicates = {
            'positive': x > 0,
            'less_than_10': x < 10
        }
        
        sfa = SFA(
            {'q0', 'q1'},
            predicates,
            {
                ('q0', 'positive'): 'q1',
                ('q0', 'less_than_10'): 'q1'
            },
            'q0',
            {'q1'},
            IntSort()
        )
        
        # Should accept as number satisfies 'positive'
        word = [IntVal(5)]
        self.assertTrue(sfa.accepts(word))

    def test_no_valid_transition(self):
        """Test case where no valid transition exists"""
        x = Int('x')
        predicates = {'positive': x > 0}
        
        sfa = SFA(
            {'q0', 'q1'},
            predicates,
            {('q0', 'positive'): 'q1'},
            'q0',
            {'q1'},
            IntSort()
        )
        
        # Should reject as -1 doesn't satisfy 'positive'
        word = [IntVal(-1)]
        self.assertFalse(sfa.accepts(word))

    def test_init_valid(self):
        """Test valid SFA initialization"""
        self.assertEqual(self.sfa.states, self.states)
        self.assertEqual(self.sfa.predicates, self.predicates)
        self.assertEqual(self.sfa.transitions, self.transitions)
        self.assertEqual(self.sfa.initial_state, self.initial_state)
        self.assertEqual(self.sfa.accepting_states, self.accepting_states)
        self.assertEqual(self.sfa.sort, self.sort)
        self.assertIsInstance(self.sfa._solver, Solver)

    def test_init_empty_states(self):
        """Test SFA initialization with empty states"""
        with self.assertRaises(ValueError) as cm:
            SFA(set(), self.predicates, self.transitions,
                self.initial_state, self.accepting_states, self.sort)
        self.assertEqual(str(cm.exception), "States set cannot be empty")

    def test_init_invalid_initial_state(self):
        """Test SFA initialization with invalid initial state"""
        with self.assertRaises(ValueError) as cm:
            SFA(self.states, self.predicates, self.transitions,
                'invalid', self.accepting_states, self.sort)
        self.assertEqual(str(cm.exception), 
                        "Initial state 'invalid' not in states")

    def test_init_invalid_accepting_states(self):
        """Test SFA initialization with invalid accepting states"""
        invalid_accepting = {'invalid', 'q1'}
        with self.assertRaises(ValueError) as cm:
            SFA(self.states, self.predicates, self.transitions,
                self.initial_state, invalid_accepting, self.sort)
        self.assertEqual(str(cm.exception),
                        "Accepting states {'invalid'} not in states")

    def test_init_non_boolean_predicate(self):
        """Test SFA initialization with non-boolean predicate"""
        x = Int('x')
        invalid_predicates = {
            'even': x % 2 == 0,
            'value': x  # Not a boolean expression
        }
        with self.assertRaises(ValueError) as cm:
            SFA(self.states, invalid_predicates, self.transitions,
                self.initial_state, self.accepting_states, self.sort)
        self.assertEqual(str(cm.exception),
                        "Predicate 'value' must be a boolean expression")

    def test_init_invalid_transition_state(self):
        """Test SFA initialization with invalid transition state"""
        invalid_transitions = {
            ('invalid', 'even'): 'q1',
            ('q0', 'odd'): 'q0'
        }
        with self.assertRaises(ValueError) as cm:
            SFA(self.states, self.predicates, invalid_transitions,
                self.initial_state, self.accepting_states, self.sort)
        self.assertEqual(str(cm.exception),
                        "Transition from invalid state 'invalid'")

    def test_init_invalid_transition_predicate(self):
        """Test SFA initialization with invalid transition predicate"""
        invalid_transitions = {
            ('q0', 'invalid'): 'q1',
            ('q0', 'odd'): 'q0'
        }
        with self.assertRaises(ValueError) as cm:
            SFA(self.states, self.predicates, invalid_transitions,
                self.initial_state, self.accepting_states, self.sort)
        self.assertEqual(str(cm.exception),
                        "Unknown predicate 'invalid'")

    def test_init_invalid_transition_next_state(self):
        """Test SFA initialization with invalid next state"""
        invalid_transitions = {
            ('q0', 'even'): 'invalid',
            ('q0', 'odd'): 'q0'
        }
        with self.assertRaises(ValueError) as cm:
            SFA(self.states, self.predicates, invalid_transitions,
                self.initial_state, self.accepting_states, self.sort)
        self.assertEqual(str(cm.exception),
                        "Transition to invalid state 'invalid'")

    def test_is_universal_without_domain_constraint(self):
        """Test is_universal without domain constraint"""
        # This SFA is not universal as it only accepts sequences ending with even numbers
        self.assertFalse(self.sfa.is_universal())

    def test_is_universal_with_domain_constraint(self):
        """Test is_universal with domain constraint"""
        x = Int('x')
        # Create an SFA that accepts all positive integers
        states = {'q0'}
        predicates = {'positive': x > 0}
        transitions = {('q0', 'positive'): 'q0'}
        
        sfa = SFA(states, predicates, transitions, 'q0', {'q0'}, self.sort)
        
        # Should be universal for positive integers
        self.assertTrue(sfa.is_universal(x > 0))

    def test_is_universal_empty_language(self):
        """Test is_universal for SFA accepting empty language"""
        # Create an SFA with no accepting states
        states = {'q0'}
        x = Int('x')
        predicates = {'any': BoolVal(True)}  # Fix: Use Z3 boolean value
        transitions = {('q0', 'any'): 'q0'}
        
        sfa = SFA(states, predicates, transitions, 'q0', set(), self.sort)
        
        # Empty language is not universal
        self.assertFalse(sfa.is_universal())

    def test_is_universal_single_state_all_accepting(self):
        """Test is_universal for SFA accepting all inputs"""
        # Create an SFA that accepts everything
        states = {'q0'}
        x = Int('x')
        predicates = {'true': BoolVal(True)}  # Fix: Use Z3 boolean value
        transitions = {('q0', 'true'): 'q0'}
        
        sfa = SFA(states, predicates, transitions, 'q0', {'q0'}, self.sort)
        
        # Should be universal
        self.assertTrue(sfa.is_universal())

    def test_is_universal_with_unreachable_states(self):
        """Test is_universal with unreachable states"""
        states = {'q0', 'q1'}
        x = Int('x')
        predicates = {'true': BoolVal(True)}  # Fix: Use Z3 boolean value
        # q1 is unreachable
        transitions = {('q0', 'true'): 'q0'}
        
        sfa = SFA(states, predicates, transitions, 'q0', {'q0', 'q1'}, self.sort)
        
        # Should still be universal as all reachable paths are accepting
        self.assertTrue(sfa.is_universal())

if __name__ == '__main__':
    unittest.main()