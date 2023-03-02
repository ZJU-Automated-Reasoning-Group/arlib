# Run regression tests
SOLVER="$(pwd)/venv/bin/python3"
OPTIONS="psmt_main.py --workers 2 --logic ALL --verbose 1 "
BENCHMARK_DIR=$1
TIMEOUT=$2

echo "Benchmark dir is ${BENCHMARK_DIR}"
echo "Timeout is ${TIMEOUT}"

echo "Running psmt"

trap "exit" INT
for file in ${BENCHMARK_DIR}/*.smt2; do
    echo "Solving ${file}"
    filename=`basename ${file}`
    gtimeout ${TIMEOUT} /usr/bin/time ${SOLVER} ${OPTIONS} ${file}
done
