#!/bin/bash
# Run regression tests
set -e  # Exit on error

# Check if benchmark directory is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <benchmark_dir> [timeout_seconds]"
    exit 1
fi

BENCHMARK_DIR=$1
TIMEOUT=${2:-60}  # Default timeout is 60 seconds if not specified

# Set up paths
SOLVER="$(pwd)/venv/bin/python3"
TOOL="bvfp_solver.py --logic "
#TOOL="lira_solver.py --workers 3 --logic "

Z3_SOLVER="$(pwd)/bin_solvers/z3"
CVC5_SOLVER="$(pwd)/bin_solvers/cvc5"

# Check if solvers exist
for solver_path in "$SOLVER" "$Z3_SOLVER" "$CVC5_SOLVER"; do
    if [ ! -f "$solver_path" ] && [ ! -f "$(which $solver_path 2>/dev/null)" ]; then
        echo "Error: Solver not found: $solver_path"
        exit 1
    fi
done

# Check if benchmark directory exists
if [ ! -d "$BENCHMARK_DIR" ]; then
    echo "Error: Benchmark directory not found: $BENCHMARK_DIR"
    exit 1
fi

# Determine timeout command based on OS
if [[ $(uname -s) == "Darwin" ]]; then
    if command -v gtimeout &> /dev/null; then
        TIMEOUT_CMD="gtimeout"
    else
        echo "Warning: gtimeout not found. Install coreutils with 'brew install coreutils'"
        echo "Continuing without timeout protection"
        TIMEOUT_CMD="timeout -k 10"  # This will fail, but we'll catch it
    fi
else
    TIMEOUT_CMD="timeout -k 10"  # Add kill signal after 10 sec if process doesn't terminate
fi

echo "Benchmark dir is ${BENCHMARK_DIR}"
echo "Timeout is ${TIMEOUT} seconds"
echo "Running solvers..."

# Handle CTRL+C gracefully
trap "echo 'Interrupted. Exiting...'; exit 1" INT

# Count files for progress display
total_files=$(find "${BENCHMARK_DIR}" -name "*.smt2" | wc -l)
current=0

# Run tests
for file in ${BENCHMARK_DIR}/*.smt2; do
    # Check if file exists (in case the glob doesn't match any files)
    if [ ! -f "$file" ]; then
        echo "No .smt2 files found in $BENCHMARK_DIR"
        exit 1
    fi
    
    current=$((current+1))
    filename=$(basename "${file}")
    echo "[${current}/${total_files}] Solving ${filename}"
    
    # Extract logic safely
    logic=$(grep -m1 '^[^;]*set-logic' "$file" | sed -n 's/.*set-logic  *\([A-Z_]*\).*/\1/p')
    if [ -z "$logic" ]; then
        echo "Warning: Could not extract logic from $file, using default"
        logic="QF_BVFP"  # Default logic as fallback
    fi
    
    OPTIONS="${TOOL}${logic}"
    
    echo "Running solver 1 (our solver)..."
    ${TIMEOUT_CMD} ${TIMEOUT} /usr/bin/time ${SOLVER} ${OPTIONS} ${file}
    
    echo "Running solver 2 (Z3)..."
    ${TIMEOUT_CMD} ${TIMEOUT} /usr/bin/time ${Z3_SOLVER} ${file}
    
    echo "Running solver 3 (CVC5)..."
    ${TIMEOUT_CMD} ${TIMEOUT} /usr/bin/time ${CVC5_SOLVER} ${file}
    
    echo "-----------------------------------------"
done

echo "All tests completed!"
