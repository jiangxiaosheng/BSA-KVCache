#!/bin/bash
set -euo pipefail

CACHE_SIZES=(0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
: "${NUM_PROCS:=$(nproc 2>/dev/null || echo 4)}"

printf "%s\n" "${CACHE_SIZES[@]}" | xargs -P "${NUM_PROCS}" -n1 -I{} bash -c '
cache_size="$1"
echo "Running s3fifo with cache size: ${cache_size} GB"
MAIN_ARGS=(--override eviction_algorithm=s3fifo --override cache_size="${cache_size}")
python3 src/simulator.py "${MAIN_ARGS[@]}" | tee "results/s3fifo_cache_size_${cache_size}_hit_rate.log"
' _ {}
