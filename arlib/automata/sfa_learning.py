"""
Learning SFA (Symbolic Finite Automata)
"""

from __future__ import print_function
import os
import sys
from typing import List, Dict, Set, Tuple, Optional, Union, Any



class SFALearner:
    """
    Base class for learning Symbolic Finite Automata (SFA).
    
    This class provides the foundation for implementing various SFA learning algorithms.
    Subclasses should implement specific learning strategies.
    """
    
    def __init__(self):
        """Initialize the SFA learner."""
        self.states = set()
        self.initial_state = None
        self.final_states = set()
        self.transitions = {}
        
    def learn(self, positive_samples: List[str], negative_samples: List[str]) -> 'SFA':
        """
        Learn an SFA from positive and negative samples.
        
        Args:
            positive_samples: List of strings that should be accepted by the automaton
            negative_samples: List of strings that should be rejected by the automaton
            
        Returns:
            An SFA that accepts the positive samples and rejects the negative samples
        """
        raise NotImplementedError("Subclasses must implement the learn method")
    
    def evaluate(self, sfa: 'SFA', samples: List[str]) -> float:
        """
        Evaluate the learned SFA on a set of samples.
        
        Args:
            sfa: The learned SFA
            samples: List of samples to evaluate
            
        Returns:
            Accuracy of the SFA on the given samples
        """
        correct = 0
        for sample in samples:
            if sfa.accepts(sample) == (sample in self.positive_samples):
                correct += 1
        return correct / len(samples) if samples else 0.0
