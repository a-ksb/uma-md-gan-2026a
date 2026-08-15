#!/usr/bin/env bash
#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM --mpi proc=4
#PJM -L elapse=24:00:00
#PJM -j

set -euo pipefail

VENV_DIR="${VENV_DIR:-$HOME/venvs/uma_oc20}"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "venv not found: $VENV_DIR" >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1

module load gcc-toolset
module load cuda/12.2.2
module load ompi-cuda
module load lammps-cuda

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# =====================================================================
# Chunk settings: edit this block before `pjsub run_oc20_fix_external_b_batch.sh`.
#
# 1st run, no resume:
#   START_STEP=1
#   END_STEP=300000
#   OUTPUT_PARENT="outputs"
#   RESUME_FROM_RUN_DIR=""
#   -> writes directly into outputs/1-300000/
#
# 2nd run, continue from the 1st run:
#   START_STEP=300001
#   END_STEP=600000
#   OUTPUT_PARENT="outputs"
#   RESUME_FROM_RUN_DIR="outputs/1-300000"
#   -> writes directly into outputs/300001-600000/
#
# 3rd run, continue from the 2nd run:
#   START_STEP=600001
#   END_STEP=900000
#   OUTPUT_PARENT="outputs"
#   RESUME_FROM_RUN_DIR="outputs/300001-600000"
#   -> writes directly into outputs/600001-900000/
#
# For later runs, set:
#   START_STEP = previous END_STEP + 1
#   END_STEP   = START_STEP + 300000 - 1
#   RESUME_FROM_RUN_DIR = previous output directory
# =====================================================================
START_STEP=1200001
END_STEP=1500000
OUTPUT_PARENT="outputs"
RESUME_FROM_RUN_DIR="outputs/900001-1200000"

# Runtime/reporting intervals.
MD_PROGRESS_INTERVAL=100
RESTART_INTERVAL_STEPS=10000

# Safety: keep 0 for production. Set to 1 only when intentionally rerunning
# the same START_STEP-END_STEP output directory and overwriting old files.
ALLOW_OVERWRITE=0

if (( START_STEP < 1 || END_STEP < START_STEP )); then
  echo "Invalid step range: START_STEP=$START_STEP END_STEP=$END_STEP" >&2
  exit 1
fi

if (( START_STEP == 1 )) && [[ -n "$RESUME_FROM_RUN_DIR" ]]; then
  echo "START_STEP=1 must start fresh, but RESUME_FROM_RUN_DIR is set: $RESUME_FROM_RUN_DIR" >&2
  exit 1
fi

if (( START_STEP > 1 )) && [[ -z "$RESUME_FROM_RUN_DIR" ]]; then
  echo "START_STEP=$START_STEP requires RESUME_FROM_RUN_DIR to continue from the previous chunk." >&2
  echo "Example for the 2nd chunk: RESUME_FROM_RUN_DIR=outputs/1-300000" >&2
  exit 1
fi

if [[ -n "$RESUME_FROM_RUN_DIR" ]]; then
  if [[ ! -d "$RESUME_FROM_RUN_DIR" ]]; then
    echo "Resume directory not found: $RESUME_FROM_RUN_DIR" >&2
    exit 1
  fi
  if [[ ! -f "$RESUME_FROM_RUN_DIR/oc20_fix_external.dump" ]]; then
    echo "Resume dump not found: $RESUME_FROM_RUN_DIR/oc20_fix_external.dump" >&2
    exit 1
  fi
  if compgen -G "$RESUME_FROM_RUN_DIR/oc20_fix_external.restart.*" > /dev/null; then
    echo "Resume mode: exact restart from the latest oc20_fix_external.restart.* in $RESUME_FROM_RUN_DIR"
  else
    echo "Warning: no oc20_fix_external.restart.* found in $RESUME_FROM_RUN_DIR; falling back to dump snapshot resume." >&2
  fi
fi

STEPS=$((END_STEP - START_STEP + 1))
RUN_DIR="${OUTPUT_PARENT}/${START_STEP}-${END_STEP}"

if [[ -e "$RUN_DIR" && "$ALLOW_OVERWRITE" != "1" ]]; then
  if [[ -f "$RUN_DIR/summary.json" || -f "$RUN_DIR/oc20_fix_external.dump" || -f "$RUN_DIR/lammps.log" ]] || compgen -G "$RUN_DIR/oc20_fix_external.restart.*" > /dev/null; then
    echo "Output directory already contains run artifacts: $RUN_DIR" >&2
    echo "Use a new START_STEP-END_STEP range or set ALLOW_OVERWRITE=1 only if rerunning intentionally." >&2
    exit 1
  fi
fi

mkdir -p "$RUN_DIR"

CMD=(
  python oc20_relax_then_lammps_nvt_workflow.py
  --optimized-structure 01_optimized_t0.vasp
  --fixed-atom-ids-file fixed_atom_ids_1based.txt
  --model-path uma-m-1p1.pt
  --output-root "$RUN_DIR"
  --output-layout flat
  --run-name "oc20_${START_STEP}_${END_STEP}"
  --temperature-k 1273.0
  --timestep-fs 0.1
  --tdamp-fs 10.0
  --thermostat-chain-length 5
  --fixed-layer-count 0
  --nvt-layer-start 1
  --nvt-layer-end 4
  --steps "$STEPS"
  --md-progress-interval "$MD_PROGRESS_INTERVAL"
  --restart-interval-steps "$RESTART_INTERVAL_STEPS"
)

if [[ -n "$RESUME_FROM_RUN_DIR" ]]; then
  CMD+=(--resume-from-run-dir "$RESUME_FROM_RUN_DIR")
fi

echo "Running chunk: ${START_STEP}-${END_STEP}"
echo "Output directory: $RUN_DIR"
if [[ -n "$RESUME_FROM_RUN_DIR" ]]; then
  echo "Resume source: $RESUME_FROM_RUN_DIR"
else
  echo "Resume source: none"
fi
echo "Command: ${CMD[*]}"

exec "${CMD[@]}"
