#!/usr/bin/env bash
set -euo pipefail

START_DATE="${START_DATE:-2026-05-01}"
EXPECTED_QOS="${EXPECTED_QOS:-embers}"

bad_scripts="$(grep -RIn --include='*.sbatch' -E '#SBATCH --qos=' scripts 2>/dev/null | awk -v expected="--qos=${EXPECTED_QOS}" '$0 !~ expected {print}')"
if [[ -n "${bad_scripts}" ]]; then
  echo "Slurm scripts with unexpected QOS:" >&2
  echo "${bad_scripts}" >&2
  exit 1
fi

if command -v squeue >/dev/null 2>&1; then
  echo "Current jobs:"
  squeue -u "${USER}" -o "%.18i %.12P %.28j %.10T %.10q %.30b %.32a"
fi

if command -v sacct >/dev/null 2>&1; then
  echo
  echo "Jobs since ${START_DATE} not using ${EXPECTED_QOS}:"
  nonmatching="$(
    sacct -u "${USER}" -S "${START_DATE}" -X \
      --format=JobID,JobName,Account,QOS,Partition,State,Elapsed -P |
      awk -F'|' -v expected="${EXPECTED_QOS}" 'NR == 1 || $4 != expected {print}'
  )"
  echo "${nonmatching}"
  if [[ "$(echo "${nonmatching}" | wc -l)" -gt 1 ]]; then
    exit 1
  fi
fi
