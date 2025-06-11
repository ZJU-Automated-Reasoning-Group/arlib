#!/usr/bin/env python3
"""Automata examples using arlib's finite and symbolic automata"""

import z3
from arlib.automata.fa import DFA, NFA
from arlib.automata.sfa import SFA

def dfa_example():
    print("=== DFA Example ===")
    states = {'q0', 'q1'}
    alphabet = {'a', 'b'}
    transitions = {('q0', 'a'): 'q0', ('q0', 'b'): 'q1', ('q1', 'a'): 'q0', ('q1', 'b'): 'q1'}
    
    dfa = DFA(states, alphabet, transitions, 'q0', {'q1'})
    
    for word in ['', 'a', 'b', 'ab', 'abb']:
        result = dfa.accepts(word)
        print(f"'{word}': {'✓' if result else '✗'}")
    
    min_dfa = dfa.minimize()
    print(f"States: {len(dfa.states)} -> {len(min_dfa.states)}")

def nfa_example():
    print("\n=== NFA Example ===")
    states = {'q0', 'q1', 'q2'}
    alphabet = {'a', 'b'}
    transitions = {
        ('q0', 'a'): {'q0', 'q1'},
        ('q0', 'b'): {'q0'},
        ('q1', ''): {'q2'},
        ('q1', 'b'): {'q2'},
        ('q2', 'b'): {'q2'}
    }
    
    nfa = NFA(states, alphabet, transitions, {'q0'}, {'q2'})
    
    for word in ['', 'a', 'ab', 'aab']:
        result = nfa.accepts(word)
        print(f"'{word}': {'✓' if result else '✗'}")
    
    dfa = nfa.to_dfa()
    print(f"NFA->DFA: {len(nfa.states)} -> {len(dfa.states)}")

def sfa_example():
    print("\n=== SFA Example ===")
    try:
        states = {'q0', 'q1'}
        x = z3.Int('x')
        predicates = {'positive': x > 0, 'zero': x == 0}
        transitions = {('q0', 'positive'): 'q1', ('q1', 'zero'): 'q1'}
        
        sfa = SFA(states, predicates, transitions, 'q0', {'q1'}, z3.IntSort())
        
        test_seqs = [[z3.IntVal(5)], [z3.IntVal(-1)], [z3.IntVal(3), z3.IntVal(0)]]
        for seq in test_seqs:
            result = sfa.accepts(seq)
            seq_str = ', '.join(str(v) for v in seq)
            print(f"[{seq_str}]: {'✓' if result else '✗'}")
            
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    dfa_example()
    nfa_example()
    sfa_example() 