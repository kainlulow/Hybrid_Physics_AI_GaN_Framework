import os
import numpy as np
import pandas as pd
import re

# === CONFIG ===
INPUT_CSV = "Dataset.csv"                      # your input file path
OUTPUT_CSV = "extracted_Id_Vgs_long.csv"       # master CSV
IV_FOLDER = "IV"                               # folder for per-sample CSV files
FEATURE_COLS = ["sde_xAl", "sde_xIn", "sde_tRecess", "sde_AlN1", "sde_AlN2"]
VGS_START, VGS_STOP, VGS_POINTS = 5, -10, 300  # as per your device setup
VDS_CONST = -5                                 # constant drain voltage

# === LOAD ===
df = pd.read_csv(INPUT_CSV)

# --- Validate required columns ---
missing_features = [c for c in FEATURE_COLS if c not in df.columns]
if missing_features:
    raise ValueError(f"Missing required feature columns: {missing_features}")
if "Nodes" not in df.columns or "SourceFolder" not in df.columns:
    raise ValueError("The input CSV must contain 'Nodes' and 'SourceFolder' columns.")

# Find Id columns I0..I299 and sort numerically
id_cols = sorted(
    [c for c in df.columns if c.startswith("I") and c[1:].isdigit()],
    key=lambda x: int(x[1:])
)
if len(id_cols) != VGS_POINTS:
    raise ValueError(
        f"Expected {VGS_POINTS} Id columns (I0..I{VGS_POINTS-1}), "
        f"but found {len(id_cols)}"
    )

# === BUILD Vgs vector ===
Vgs_vec = np.linspace(VGS_START, VGS_STOP, VGS_POINTS)
vgs_map = {i: Vgs_vec[i] for i in range(VGS_POINTS)}

# === RESHAPE TO LONG FORMAT ===
df = df.reset_index(drop=True).reset_index(names="row_id")
subset = df[["row_id"] + FEATURE_COLS + ["Nodes", "SourceFolder"] + id_cols]

melted = subset.melt(
    id_vars=["row_id"] + FEATURE_COLS + ["Nodes", "SourceFolder"],
    value_vars=id_cols,
    var_name="Icol",
    value_name="Ids"
)
melted["I_index"] = melted["Icol"].str.replace("I", "", regex=False).astype(int)
melted = melted.sort_values(["row_id", "I_index"], kind="mergesort")
melted["Vgs"] = melted["I_index"].map(vgs_map).astype(float)
melted["Vds"] = float(VDS_CONST)

# === FINAL COLUMN ORDER (master CSV) ===
final = melted[FEATURE_COLS + ["Vds", "Vgs", "Ids"]]
final.to_csv(OUTPUT_CSV, index=False)
print(f"Master CSV saved to: {OUTPUT_CSV}")

# === SAVE INDIVIDUAL CSV FILES ===
os.makedirs(IV_FOLDER, exist_ok=True)

# For splitting we keep Nodes and SourceFolder
melted_full = melted[["row_id", "Nodes", "SourceFolder"] + FEATURE_COLS + ["Vds", "Vgs", "Ids"]]

for row_id, group_df in melted_full.groupby("row_id"):
    nodes = group_df["Nodes"].iloc[0]
    src_folder = group_df["SourceFolder"].iloc[0]

    # Clean SourceFolder for safe filenames
    safe_src_folder = re.sub(r'[^A-Za-z0-9_-]', '_', str(src_folder))
    out_file = os.path.join(IV_FOLDER, f"sample_{nodes}_{safe_src_folder}.csv")

    sample_df = group_df[FEATURE_COLS + ["Vds", "Vgs", "Ids"]]
    sample_df.to_csv(out_file, index=False)

print(f"Saved individual IV CSV files in folder: {IV_FOLDER}")
