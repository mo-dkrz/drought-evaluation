#!/bin/bash
#SBATCH --job-name=mk_ensemble
#SBATCH --qos=short
#SBATCH --account=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/mk_ensemble_%j.log
#SBATCH --error=logs/mk_ensemble_%j.err

set -e
set -u
set -o pipefail

mkdir -p logs

# =============================================================================
# Paths — must match batch_drought.sh
# =============================================================================

OUT_DIR="${HOME}/drought_catalog"
SCRIPTS_DIR="${HOME}/drought-evaluation"
PYTHON="${HOME}/drought-pipeline/bin/python"

# =============================================================================
# MK ensemble aggregation — all 3 scenarios
# input:  mk_trends_{model}_{ssp}.nc
# Output: mk_ensemble_{ssp}_isimip3b.nc
# =============================================================================

echo "======================================"
echo "MK Ensemble Aggregation"
echo "Start: $(date)"
echo "======================================"

for SSP in ssp126 ssp370 ssp585; do
    ENS_OUT="${OUT_DIR}/mk_ensemble_${SSP}_isimip3b.nc"
    if [ -f "${ENS_OUT}" ]; then
        echo "Already exists: ${ENS_OUT} — skipping."
        continue
    fi
    echo ""
    echo "--- Scenario: ${SSP} ---"
    ${PYTHON} ${SCRIPTS_DIR}/ensemble_mk_trends.py \
      --ssp     ${SSP} \
      --mk-dir  ${OUT_DIR} \
      --out-dir ${OUT_DIR}
done

echo ""
echo "======================================"
echo "Done at $(date)"
echo "======================================"
