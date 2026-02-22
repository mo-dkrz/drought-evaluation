# Drought Event Detection Pipeline

Detects and catalogs drought events from monthly SPEI output data (ISIMIP).


## Setup

Install dependencies with conda or pip:

```bash
git clone https://github.com/mo-dkrz/drought-evaluation.git
cd drought-evaluation
module load anaconda
conda create --prefix $HOME/drought-pipeline -c conda-forge python scipy numpy pandas xarray netcdf4 matplotlib cartopy dask tqdm pymannkendall -y
conda activate $HOME/drought-pipeline

DATA_DIR="$HOME/cartopy_data"
mkdir -p "$DATA_DIR"
$HOME/drought-pipeline/bin/cartopy_feature_download gshhs physical cultural cultural-extra -o "$DATA_DIR" --no-warn --ignore-repo-data
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

python compute_return_periods.py \
  --model       gfdl-esm4 \
  --ssp         ssp126 \
  --catalog-dir ~/drought_catalog \
  --out-dir     ~/drought_catalog

python compute_mk_trends.py \
  --model     gfdl-esm4 \
  --ssp       ssp126 \
  --input-dir ~/spei_r_outputs \
  --out-dir   ~/drought_catalog
```

## Running on the HPC (all 15 model/scenario combinations)

Edit the paths at the top of `batch_drought.sh`:

```bash
SPEI_DIR="${HOME}/spei_r_outputs_penman_1850_2015"
OUT_DIR="${HOME}/drought_catalog"
SCRIPTS_DIR="${HOME}/drought-evaluation"
PYTHON="${HOME}/drought-pipeline/bin/python"
```

Submit the main array job, then the ensemble job with a SLURM dependency
so it only starts once all 15 tasks are finished:

```bash
JID=$(sbatch --parsable batch_drought.sh)
sbatch --dependency=afterok:${JID} batch_mk_ensemble.sh

# Monitor
squeue -u $USER
tail -f logs/drought_<jobid>_0.log
```
