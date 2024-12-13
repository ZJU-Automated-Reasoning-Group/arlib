# source ./bin/activate
#!/usr/bin/env bash
set -euo pipefail
bench="$1"
logic=$(expr "$(grep -m1 '^[^;]*set-logic' "$bench")" : ' *(set-logic  *\([A-Z_]*\) *) *$')

case "$logic" in

QF_BV)
  python3 bvfp_solver.py --logic QF_BV $1
  ;;
QF_UFBV)
  python3 bvfp_solver.py --logic QF_UFBV $1
  ;;
QF_ABV)
  python3 bvfp_solver.py --logic QF_ABV $1
  ;;
QF_AUFBV)
  python3 bvfp_solver.py --logic QF_AUFBV $1
  ;;
QF_FP)
  python3 bvfp_solver.py --logic QF_FP $1
  ;;
QF_BVFP)
  python3 bvfp_solver.py --logic QF_BVFP $1
  ;;
*)
  # just run the default
  python3 bvfp_solver.py $1
  ;;
esac