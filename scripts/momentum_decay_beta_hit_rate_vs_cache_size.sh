#!/bin/bash
set -euo pipefail

CACHE_SIZES=(0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5)
BETAS=(0.5 0.8 0.9 0.95 0.99)
: "${NUM_PROCS:=$(nproc 2>/dev/null || echo 4)}"

mkdir -p results/decay_beta

# Build beta,cache_size pairs and launch them in parallel
COMBOS=()
for beta in "${BETAS[@]}"; do
  for cache_size in "${CACHE_SIZES[@]}"; do
    COMBOS+=("${beta},${cache_size}")
  done
done

printf "%s\n" "${COMBOS[@]}" | xargs -P "${NUM_PROCS}" -n1 -I{} bash -c '
  combo="$1"
  beta="${combo%%,*}"
  cache_size="${combo#*,}"
  log_dir="results/decay_beta"
  log_file="${log_dir}/momentum_decay_beta_${beta}_cache_size_${cache_size}_hit_rate.log"
  echo "Running momentum_decay with beta=${beta}, cache size: ${cache_size} GB"
  MAIN_ARGS=(--override eviction_algorithm=momentum_decay --override cache_size="${cache_size}" --override momentum_beta="${beta}")
  python3 src/simulator.py "${MAIN_ARGS[@]}" | tee "${log_file}"
' _ {}
