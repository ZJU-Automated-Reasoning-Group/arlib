#!/bin/bash
# Run regression tests
SOLVER="$(pwd)/venv/bin/python3"
TOOL="bvfp_solver.py --logic "
#TOOL="lira_solver.py --workers 3 --logic "
BENCHMARK_DIR=$1
TIMEOUT=$2

Z3_SOLVER="$(pwd)/bin_solvers/z3"
CVC5_SOLVER="$(pwd)/bin_solvers/cvc5"

if [[ $(uname -s) == "Darwin" ]]; then
  TIMEOUT_CMD="gtimeout"
else
  TIMEOUT_CMD="timeout"
fi

echo "Benchmark dir is ${BENCHMARK_DIR}"
echo "Timeout is ${TIMEOUT}"

echo "Running solvers"

trap "exit" INT
for file in ${BENCHMARK_DIR}/*.smt2; do
    echo "Solving ${file}"
    filename=`basename ${file}`
    logic=$(expr "$(grep -m1 '^[^;]*set-logic' "$file")" : ' *(set-logic  *\([A-Z_]*\) *) *$')
    OPTIONS=$TOOL$logic
    # echo ${OPTIONS}
    # TODO: gtimeout is for Mac..
    ${TIMEOUT_CMD} ${TIMEOUT} /usr/bin/time ${SOLVER} ${OPTIONS} ${file}
    ${TIMEOUT_CMD} ${TIMEOUT} /usr/bin/time ${Z3_SOLVER} ${file}
    ${TIMEOUT_CMD} ${TIMEOUT} /usr/bin/time ${CVC5_SOLVER}  ${file}
done
