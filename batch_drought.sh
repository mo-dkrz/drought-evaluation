#!/bin/bash
#SBATCH --job-name=drought_catalog
#SBATCH --array=0-14
#SBATCH --qos=short
#SBATCH --account=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/drought_%A_%a.log
#SBATCH --error=logs/drought_%A_%a.err

set -e
set -u
set -o pipefail

mkdir -p logs

# =============================================================================
# Input configuration
# =============================================================================

SPEI_DIR="${HOME}/spei_r_outputs_penman_1850_2015"
OUT_DIR="${HOME}/drought_catalog"
SCRIPTS_DIR="${HOME}/drought-evaluation"
PYTHON="${HOME}/drought-pipeline/bin/python"

BASELINE_START="1979-01-01"
BASELINE_END="2014-12-31"
PERCENTILE="0.05"
SPELL_REF="-0.4"
ZOOM_START="1980-01-01"
ZOOM_END="2014-12-31"
HIST_START="1950-01-01"
HIST_END="2014-12-31"
FAR_START="2036-01-01"
FAR_END="2100-12-31"
RETURN_PERIODS="2,5,10,20,50,100"
LOCATIONS=(
    "Atlanta:33.7:-84.4"
    "Lawrence:39.0:-95.2"
    "Phoenix:33.4:-112.0"
    "Seattle:47.6:-122.3"
    "Ithaca:42.4:-76.5"
)


# =============================================================================
# MAP TASK ID -> model / scenario
# =============================================================================
mkdir -p $OUT_DIR

MODELS=("gfdl-esm4" "ukesm1-0-ll" "mpi-esm1-2-hr" "ipsl-cm6a-lr" "mri-esm2-0")
SCENARIOS=("ssp126" "ssp370" "ssp585")

MODEL="${MODELS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SCENARIO="${SCENARIOS[$((SLURM_ARRAY_TASK_ID % 3))]}"

echo "======================================"
echo "Task ${SLURM_ARRAY_TASK_ID}: ${MODEL} / ${SCENARIO}"
echo "Start: $(date)"
echo "======================================"

# =============================================================================
# CHECK INPUT EXISTS
# =============================================================================

SPEI_FILE="${SPEI_DIR}/${MODEL}_${SCENARIO}/spei_${MODEL}_${SCENARIO}.nc"
if [ ! -f "${SPEI_FILE}" ]; then
    echo "ERROR: SPEI input not found: ${SPEI_FILE}"
    echo "Run SPEI script first."
    exit 1
fi
echo "Input: ${SPEI_FILE}  ($(du -h ${SPEI_FILE} | cut -f1))"

# =============================================================================
# SKIP IF ALREADY DONE
# =============================================================================

CATALOG="${OUT_DIR}/drought_event_catalog_${MODEL}_${SCENARIO}.nc"
if [ -f "${CATALOG}" ]; then
    echo "Already exists: ${CATALOG} — skipping."
    exit 0
fi

# =============================================================================
# RUN PIPELINE
# =============================================================================

COMMON_ARGS="
  --model          ${MODEL}
  --ssp            ${SCENARIO}
  --input-dir      ${SPEI_DIR}
  --out-dir        ${OUT_DIR}
  --baseline-start ${BASELINE_START}
  --baseline-end   ${BASELINE_END}
  --percentile     ${PERCENTILE}
"

echo ""
echo "--- Step 1: Compute thresholds ---"
${PYTHON} ${SCRIPTS_DIR}/compute_thershold.py ${COMMON_ARGS}

echo ""
echo "--- Step 2: Detect events ---"
${PYTHON} ${SCRIPTS_DIR}/detect_events.py ${COMMON_ARGS} --spell-ref ${SPELL_REF}

# =============================================================================
# QUICK VALIDATION
# =============================================================================

echo ""
echo "--- Output ---"
if [ -f "${CATALOG}" ]; then
    echo "File: ${CATALOG}  ($(du -h ${CATALOG} | cut -f1))"
    ${PYTHON} -c "
import xarray as xr, numpy as np
ds = xr.open_dataset('${CATALOG}')
n  = ds.sizes['event']
print(f'  Events total      : {n}')
if n > 0:
    dur = ds['duration_months'].values
    dv  = ds['deficit_volume'].values
    mn  = ds['min_spei'].values
    print(f'  Duration (months) : min={dur.min()}, median={np.median(dur):.1f}, max={dur.max()}')
    print(f'  Deficit volume    : min={dv.min():.3f}, median={np.median(dv):.3f}, max={dv.max():.3f}')
    print(f'  Min SPEI          : min={mn.min():.3f}, median={np.median(mn):.3f}')
ds.close()
"
else
    echo "ERROR: Catalog not created!"
    exit 1
fi

echo ""
echo "--- Step 3: Diagnostics ---"
${PYTHON} ${SCRIPTS_DIR}/diagnostics.py \
  --model          ${MODEL} \
  --ssp            ${SCENARIO} \
  --input-dir      ${SPEI_DIR} \
  --out-dir        ${OUT_DIR} \
  --percentile     ${PERCENTILE} \
  --spell-ref      ${SPELL_REF} \
  --zoom-start     ${ZOOM_START} \
  --zoom-end       ${ZOOM_END} \
  --locations      "${LOCATIONS[@]}"


echo ""
echo "--- Step 4: Return periods (all grid cells) ---"
RP_OUT="${OUT_DIR}/return_periods_${MODEL}_${SCENARIO}.nc"
if [ -f "${RP_OUT}" ]; then
    echo "Already exists: ${RP_OUT} — skipping."
else
    ${PYTHON} ${SCRIPTS_DIR}/compute_return_periods.py \
      --model          ${MODEL} \
      --ssp            ${SCENARIO} \
      --catalog-dir    ${OUT_DIR} \
      --out-dir        ${OUT_DIR} \
      --return-periods ${RETURN_PERIODS} \
      --hist-start     ${HIST_START} \
      --hist-end       ${HIST_END} \
      --far-start      ${FAR_START} \
      --far-end        ${FAR_END}
fi

echo ""
echo "======================================"
echo "Done: ${MODEL} / ${SCENARIO}  at $(date)"
echo "======================================"