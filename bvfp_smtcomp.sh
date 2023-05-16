# Run regression tests
PYTHON="$(pwd)/venv/bin/python3"
TOOL="bvfp_solver.py --logic "
BENCHMARK=$1


trap "exit" INT
LOGIC=$(expr "$(grep -m1 '^[^;]*set-logic' "$BENCHMARK")" : ' *(set-logic  *\([A-Z_]*\) *) *$')
OPTIONS=$TOOL$LOGIC
# echo "Solving ${BENCHMARK}"
${PYTHON} ${OPTIONS} ${BENCHMARK}

# TODO: set the sat engine base on logic type

