# SMT Solving for Finite Field Arithmetic

This module implements SMT solving capabilities for finite field arithmetic, supporting various theories and solving techniques.

## Overview

The finite field SMT solver supports:
- Quantifier-free finite field formulas (QF_FF or QF_FFA)


## Features

### Core Capabilities
- Native support for finite field operations (add, mul, div)
- Translation of finite field formulas to bit-vector (QF_BV) or integer arithmetic (QF_NIA)

### Advanced Features
- Gröbner basis computation for algebraic reasoning (via CVC5)
- MCSat-based solving techniques (via Yices2)
