# Drought Event Detection Pipeline

Detects and catalogs drought events from monthly SPEI output data (ISIMIP).


## Setup

Install dependencies with conda or pip:

```bash
module load anaconda
conda create --prefix $HOME/drought-pipeline -c conda-forge python numpy pandas xarray netcdf4 matplotlib cartopy -y
conda activate drought
```


## Running locally (single model/scenario)

```bash
python compute_thershold.py \
    --model gfdl-esm4 --ssp ssp126 \
    --input-dir ~/spei_r_outputs \
    --out-dir   ~/drought_catalog

python detect_events.py \
    --model gfdl-esm4 --ssp ssp126 \
    --input-dir ~/spei_r_outputs \
    --out-dir   ~/drought_catalog

python diagnostics.py \
    --model gfdl-esm4 --ssp ssp126 \
    --input-dir ~/spei_r_outputs \
    --out-dir   ~/drought_catalog
```

## Running on the HPC (all 15 model/scenario combinations)

Edit the paths at the top of `batch_drought.sh`:

```bash
SPEI_DIR="${HOME}/spei_r_outputs"
OUT_DIR="${HOME}/drought_catalog"
SCRIPTS_DIR="${HOME}/drought-evaluation"
PYTHON="${HOME}/drought-pipeline/bin/python"
```

Then submit:

```bash
sbatch batch_drought.sh

# Monitor
squeue -u $USER
tail -f logs/drought_<jobid>_0.log
```

