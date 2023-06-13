# Model Counting

## Introduction to Model Counting
Model counting is the problem of determining the number of possible solutions
(models) to a given formula. It is a fundamental problem in computer 
science and has applications in various fields such as artificial intelligence, cryptography, and verification.


## Model Counting in Arlib

### Model Counting for SAT Formulas

To count models for SAT formulas, please use `sharpSAT` or other third-party tools.

### Model Counting for QF_BV Formulas

QF_BV stands for quantifier-free bit-vector logic. It is a subset of the SMT-LIB standard and is commonly used in the analysis and verification of computer hardware and software systems. 

To count the models of a QF_BF formula, refer to 
- `arlib\bv\qfbv_counting.py`.
- `arlib\tests\test_bv_counting.py`

Note that we rely on sharpSAT for the implementation. Currently, you need to either copy a 
binary version of sharpSAT to `bin_solvers` or install a sharpSAT globally.

## References