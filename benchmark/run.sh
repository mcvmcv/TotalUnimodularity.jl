#!/bin/bash
# CMR vs is_totally_unimodular on the matrices of benchmark/generate.jl.
#
#     CMR_TU=/path/to/cmr/build/cmr-tu benchmark/run.sh [OUTDIR]
#
# Run from the repository root. Prints one tab-separated line per matrix:
# name, size, CMR answer, CMR seconds, Julia answer, Julia seconds. CMR's time
# is the recognition time it reports with --stats (no process start, no file
# reading); Julia's is measured after JIT warmup. TIMEOUT (default 180) is the
# per-matrix limit in seconds for each side; on the Julia side it includes
# startup and warmup, a second or two, since every matrix gets a fresh
# process.
set -u
CMR_TU=${CMR_TU:-cmr-tu}
TIMEOUT=${TIMEOUT:-180}
OUT=${1:-$(mktemp -d)}

julia --project=. benchmark/generate.jl "$OUT" || exit 1
# Warmup: one matrix for each route through is_totally_unimodular.
W="$OUT/m01.mat $OUT/m04.mat $OUT/m06.mat $OUT/m10.mat $OUT/m16.mat $OUT/m26.mat"

while IFS=$'\t' read -r f name size; do
  out=$(timeout -k 5 "$TIMEOUT" "$CMR_TU" "$f" --stats 2>&1)
  if [ $? -eq 124 ]; then
    cres=timeout; ct=">$TIMEOUT"
  else
    case "$out" in
      *"IS totally"*) cres=true ;;
      *"IS NOT"*)     cres=false ;;
      *)              cres="?" ;;
    esac
    ct=$(echo "$out" | awk '/seymour total:|camion total:/ {s+=$(NF-1)} END {printf "%.6f", s}')
  fi
  jo=$(timeout -k 5 "$TIMEOUT" julia --project=. benchmark/time_julia.jl "$f" $W 2>/dev/null | tail -1)
  [ -z "$jo" ] && jo=$'timeout\t>'"$TIMEOUT"
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$size" "$cres" "$ct" "$jo"
done < "$OUT/manifest.tsv"
