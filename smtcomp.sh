#!/bin/bash

solver=$(dirname "$(readlink -f "$0")")/???
bench="$1"

# Output other than "sat"/"unsat" is either written to stderr or to "err.log"
# in the directory specified by $2 if it has been set (e.g. when running on
# StarExec).
out_file=/dev/stderr

if [ -n "$STAREXEC_WALLCLOCK_LIMIT" ]; then
  # If we are running on StarExec, don't print to `/dev/stderr/` even when $2
  # is not provided.
  out_file="/dev/null"
fi

if [ -n "$2" ]; then
  out_file="$2/err.log"
fi

logic=$(expr "$(grep -m1 '^[^;]*set-logic' "$bench")" : ' *(set-logic  *\([A-Z_]*\) *) *$')
