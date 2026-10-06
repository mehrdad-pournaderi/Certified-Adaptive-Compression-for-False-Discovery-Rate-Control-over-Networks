#!/usr/bin/env bash
# Full reproduction: tests -> experiments -> figures/tables.
# Run from the repository root. Experiments take a few hours; skip them with
#   SKIP_EXPERIMENTS=1 bash reproduce.sh
# to regenerate figures/tables from the shipped sim/results/*.npz.
set -euo pipefail
cd "$(dirname "$0")"

echo "== tests"
for t in cert mono fdr ident; do python3 -m sim.journal_tests "$t"; done

if [ -z "${SKIP_EXPERIMENTS:-}" ]; then
  echo "== experiments"
  for pi in 0.01 0.03 0.1 0.3; do python3 -m sim.journal_experiments_r5 S1 "$pi"; done
  for S in 6 8 10 12 16 20;    do python3 -m sim.journal_experiments_r5 S2 "$S";  done
  for M in 3 4 5;              do python3 -m sim.journal_experiments_r5 S3 "$M";  done
  python3 -m sim.journal_experiments_r5 S3u
  python3 -m sim.journal_experiments_r5 S3n
  for v in fixed redirect_abstain fixed_levelcorr; do python3 -m sim.journal_experiments_r5 S4 "$v"; done
  for d in 2 4 8 16;           do python3 -m sim.journal_experiments_r5 S5 "$d";  done
  for B in 8 10 14;            do python3 -m sim.journal_experiments_r5 S6 "$B";  done
  python3 -m sim.journal_experiments_r5 S7
fi

echo "== figures and tables"
python3 -m sim.journal_plots_r5

echo "figures and tables written to output/"
