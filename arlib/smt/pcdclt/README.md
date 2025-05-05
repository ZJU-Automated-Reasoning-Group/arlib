# Parallel CDCL(T) SMT Solver

This module implements a parallel CDCL(T) (Conflict-Driven Clause Learning with Theory) SMT solver. It combines Boolean satisfiability solving with theory reasoning in a parallel, process-based approach.

## Overview

The CDCL(T) algorithm extends the CDCL Boolean SAT solving approach with theory reasoning:

1. Create a Boolean abstraction of an SMT formula
2. Solve the Boolean abstraction with a SAT solver
3. Check if the Boolean models satisfy the theory constraints (T-consistency)
4. Learn from conflicts and add blocking clauses
5. Repeat until finding a theory-consistent model or proving unsatisfiability

## Key Components

- `CDCLTSolver`: Abstract base class for CDCL(T) solvers
- `SequentialCDCLTSolver`: A sequential implementation
- `ParallelCDCLTSolver`: A process-based parallel implementation
- `SMTPreprocessor4Process`: Handles preprocessing SMT formulas
- `BooleanFormulaManager` and `TheoryFormulaManager`: Manage Boolean and theory formulas
- `SMTLibTheorySolver`: Interface to external theory solvers (like Z3)

## SMT Query Logging

The module now supports logging and dumping the SMT queries processed by each theory solver. This is useful for:

- Debugging theory solver interactions
- Performance profiling
- Identifying patterns in theory queries
- Analyzing unsat cores

### Configuration

Logging configuration can be found in `cdclt_config.py`:

```python
# SMT Query Logging
LOG_SMT_QUERIES = True         # Set to False to disable logging
SMT_LOG_DIR = os.path.join(os.getcwd(), "smt_query_logs")  # Default log directory
SMT_LOG_QUERY_CONTENT = True   # Log the full content of SMT queries
SMT_LOG_QUERY_RESULTS = True   # Log the results of SMT queries
SMT_LOG_ASSUMPTIONS = True     # Log assumptions for check-sat-assuming calls
```

### Log Directory Structure

When logging is enabled, the following directory structure is created:

```
smt_query_logs/
└── run_YYYYMMDD_HHMMSS/
    ├── worker_0_query_0.smt2      # First query from worker 0
    ├── worker_0_query_1.smt2      # Second query from worker 0
    ├── worker_0_summary.log       # Summary log for worker 0
    ├── worker_1_query_0.smt2      # First query from worker 1
    └── ...
```

Each query file contains:
- Metadata about the query (worker ID, timestamp, etc.)
- The actual SMT query content (if enabled)
- Assumptions used (if applicable and enabled)
- The result of the query (if enabled)

The summary log files contain statistics about all queries processed by each worker.

### Usage Example

The logging is enabled by default. To use the logs for debugging or analysis:

1. Run your SMT solver
2. Inspect the generated logs in the `smt_query_logs` directory
3. Use the logs to analyze theory solver behavior

To disable logging, set `LOG_SMT_QUERIES = False` in `cdclt_config.py`.

## Performance Considerations

Note that enabling logging, especially with `SMT_LOG_QUERY_CONTENT = True`, may affect performance during solving due to I/O operations. For maximum performance in production, consider disabling logging or limiting what is logged. 