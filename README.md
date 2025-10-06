## Hybrid Physics–AI GaN Framework

This project provides a reproducible pipeline to extract, analyze, cluster, and interpret TCAD Id–Vg data for GaN-based devices. It fuses physics-derived metrics and data-driven methods (PCA, KMeans, SHAP) to discover device behavior patterns and quantify feature importance.

## Overview

The system ingests TCAD simulation outputs, computes robust device metrics, constructs uniform Id–Vg representations, and produces analysis artifacts:

- Data extraction and metric computation from TCAD PLT/gtree outputs
- Resampling Id–Vg curves onto a common grid for comparability
- Feature engineering with PCA (shape features) and magnitude/metric fusion
- Unsupervised clustering with optimal k selection via silhouette score
- Rich visualizations of IV curves and cluster-wise statistics
- SHAP-based explainability for selected targets using array-defined inputs

## Quick Start Guide

1. Install prerequisites
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Ensure your TCAD outputs are available
   - Folder(s) like `../Simulation_Data/plt1`, `../Simulation_Data/plt2`, ...
   - Each `plt*` contains `gtree.dat` and `IdVg_n{node}_des.plt` files

3. Run the pipeline (from repo root)
   ```bash
   # install Python deps
   make install

   # extract (builds Data_Extraction/Dataset.csv) – adjust --input-path if needed
   make extract

   # reshape to long CSV and split per-sample IV files
   make reshape

   # cluster analysis and plots
   make cluster

   # SHAP analysis and plots
   make shap
   ```

4. View results
   - `Data_Extraction/Dataset.csv` and `Data_Extraction/IV/*`
   - `Clustering/IV_Clusters/*` (plots + CSVs)
   - `SHAP_Analysis/SHAP_FromArrays/*`

## Detailed Instructions

### Preparing Input Data

- Place your TCAD outputs under a root directory (default: `../Simulation_Data`).
- Expected structure:
  - Subfolders named `plt{index}` (e.g., `plt1`, `plt2`, ...).
  - Each subfolder contains `gtree.dat` and multiple `IdVg_n{node}_des.plt`.
- If you already have a combined dataset, you can skip extraction and run clustering directly with `--data-path`.

### Running Extraction

```bash
python Data_Extraction/main.py \
  --input-path ../Simulation_Data \
  --output-path Data_Extraction \
  --outfile Dataset.csv \
  --mostype p \
  --points 300 \
  --plt-start 1 --plt-end 3

# see all options
python Data_Extraction/main.py --help
```

Key options:
- `--mostype {n,p}`: Device type (sets default Vgs range and Vdd).
- `--v0 --v1 --points`: Vgs sweep definition and resampling resolution.
- `--ith-abs --vdd --vov-on-abs --vov-off-abs --ioff-bias-v`: Metric extraction parameters.
- `--plt-start --plt-end --plt-prefix`: Which `plt*` folders to include.

### Reshaping and Splitting

```bash
python Data_Extraction/IV_ML.py \
  --input-csv Data_Extraction/Dataset.csv \
  --output-csv Data_Extraction/extracted_Id_Vgs_long.csv \
  --iv-folder Data_Extraction/IV \
  --feature-cols sde_xAl,sde_xIn,sde_tRecess,sde_AlN1,sde_AlN2 \
  --vgs-start 5 --vgs-stop -10 --vgs-points 300 --vds-const -5

# see all options
python Data_Extraction/IV_ML.py --help
```

### Clustering

```bash
python Clustering/IV_clustering.py \
  --data-path Data_Extraction/Dataset.csv \
  --output-dir Clustering/IV_Clusters \
  --mostype p \
  --points 300 \
  --min-k 6 --max-k 30 \
  --pca-threshold 0.95

# see all options
python Clustering/IV_clustering.py --help
```

Outputs:
- `Clustered_IV_Data.csv` with a `Cluster` column
- PCA variance plots, silhouette-vs-k, per-cluster IV figures (linear/log)
- Per-cluster IV CSVs and `Cluster_Metrics_Summary.csv`

### SHAP Analysis

- Configure targets and input arrays in `SHAP_Analysis/SHAP_Analysis.py`:
  - `INPUT_FEATURE_ARRAYS`
  - `FORCED_TARGETS` (or rely on aliases)
  - `FILE_PATH` (defaults to `../Data_Extraction/Dataset.csv`)
- Run:
  ```bash
  (cd SHAP_Analysis && python SHAP_Analysis.py)
  ```
- Outputs include per-target SHAP summaries, plots, and per-simulation exports in `SHAP_Analysis/SHAP_FromArrays/`.

## Interpreting Results

- `Data_Extraction/Dataset.csv`:
  - Metadata (`Nodes`, `SourceFolder`) + resampled `I0..I{N-1}`
  - Metrics: `Vth`, `Ion_eq`, `Ion_fixedV`, `Ioff_eq`, `Ioff_fixed0`, `ON_OFF_eq`, `ON_OFF_fixedV`, `SS`, etc.
- Clustering:
  - Optimal PCA components chosen to reach the variance threshold
  - Best k via silhouette score
  - Plots: PCA cumulative variance, silhouette vs k, per-cluster IV curves, all-clusters overlay
  - `Cluster_Metrics_Summary.csv` aggregates metrics by cluster
- SHAP:
  - Per-target signed mean SHAP (positive/negative) with conditional feature means
  - Model metrics in original units (R², MAE)
  - Per-simulation SHAP exports (by cluster and combined)

## Customizing Verification Criteria

- Tune extraction thresholds and sweep definitions via `Data_Extraction/main.py` CLI flags.
- Adjust clustering behavior with `--min-k/--max-k`, `--pca-threshold`, and feature construction in `Clustering/IV_clustering.py`.
- Control SHAP feature sets and targets in `SHAP_Analysis/SHAP_Analysis.py` via `INPUT_FEATURE_ARRAYS` and `TARGET_NAME_ALIASES`.

## Documentation

- Metrics and resampling utilities: `Data_Extraction/mylibrary.py`
- Clustering workflow and exports: `Clustering/IV_clustering.py`
- SHAP analysis configuration: `SHAP_Analysis/SHAP_Analysis.py`
- Reproducible commands: `Makefile`

## Directory Structure

```
Hybrid_Physics_AI_GaN_Framework/
├── Data_Extraction/
│   ├── main.py                 # Extraction CLI (from TCAD PLT/gtree)
│   ├── IV_ML.py                # Reshaping and per-sample IV export CLI
│   ├── mylibrary.py            # Metrics and resampling utilities
│   ├── Dataset.csv             # Combined dataset (generated or provided)
│   └── IV/                     # Per-sample IV CSVs (generated)
├── Clustering/
│   ├── IV_clustering.py        # Clustering CLI, plots, and CSV exports
│   ├── mylibrary.py            # (legacy/auxiliary utilities)
│   └── IV_Clusters/            # Outputs: plots, CSVs, summaries
├── SHAP_Analysis/
│   ├── SHAP_Analysis.py        # SHAP explainability pipeline
│   ├── Dataset.csv             # (optional copy/reference for SHAP)
│   └── SHAP_FromArrays/        # SHAP outputs (generated)
├── TCAD/                       # Example TCAD inputs (for reference)
│   ├── gtree.dat
│   ├── IdVg_des.cmd
│   └── *.par
├── requirements.txt
├── Makefile
└── README.md
```

## Key Features

### End-to-end Pipeline
- Automated extraction → reshaping → clustering → explainability with consistent CLIs.

### Robust Metric Extraction
- Threshold-based `Vth`, equal-overdrive `Ion/Ioff`, fixed-bias `Ioff@0V`, ON/OFF ratios, `SS`.

### Shape + Magnitude Fusion
- PCA on normalized `log10(|Id|)` for shape; magnitude and metrics appended for discriminative clustering.

### Optimal K Discovery
- Silhouette-based model selection over a user-defined k-range.

### Rich Visualization and Exports
- Per-cluster and aggregate IV plots; cluster-wise metric summaries; CSV exports for downstream use.

## Requirements

- Python 3.9+
- Packages pinned in `requirements.txt`
- Optional: TCAD environment to generate PLT/gtree inputs (not required if using an existing `Dataset.csv`)

## Troubleshooting

### Common Issues

1. No data collected during extraction
   - Verify `--input-path` and that `plt*` directories contain `gtree.dat` and `IdVg_n{node}_des.plt`.
   - Adjust `--plt-start/--plt-end` and `--plt-prefix`.

2. Missing feature columns in reshaping
   - Update `--feature-cols` to match your dataset columns.

3. No IV columns found / wrong count
   - Ensure `I0..I{N-1}` columns exist and match `--vgs-points`.

4. Clustering too slow or memory-heavy
   - Reduce `--max-k`, use a subset of data, or lower `--points`.

5. Headless plotting
   - Plots are saved to files; ensure write permissions to output directories.

### Getting Help

Open an issue with:
- Command used and full CLI flags
- Minimal snippet of input data structure (column names, sample rows)
- Error trace and environment details (Python version, OS)

## License

Specify your project license here (e.g., MIT, Apache-2.0).

## Citation

If you use this framework in your research, please cite:

```
@software{hybrid_physics_ai_gan_framework,
  author = {Your Name},
  title  = {Hybrid Physics–AI GaN Framework},
  year   = {2025},
  url    = {https://github.com/your-org/Hybrid_Physics_AI_GaN_Framework}
}
```
