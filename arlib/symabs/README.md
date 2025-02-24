# Symbolic Abstraction

(Closely related to arlib/optimization)

This module implements various symbolic abstraction techniques for program analysis and verification. The main components are:

## Components

### ai_symabs: Abstract Interpretation-based Symbolic Abstraction
- Implements classic abstract interpretation domains (intervals, signs, octagons)
- Based on the bilateral algorithm from Tharkur's PhD thesis
- Supports reduced product of abstract domains

### mcai: Model Counting-based Abstract Interpretation
- Combines model counting with abstract interpretation
- Computes precision metrics for different abstract domains
- Supports analysis of bit-vector formulas

### omt_symabs: Optimization Modulo Theory-based Symbolic Abstraction  
- Uses OMT solving for computing optimal abstractions
- Supports both linear integer/real arithmetic (LIA/LRA) and bit-vectors
- Implements interval, zone and octagon abstractions
- 
### predicate_abstraction: Classic Predicate Abstraction
- Based on the CAV'06 paper "SMT Techniques for Fast Predicate Abstraction"

## Usage Examples

TBD

