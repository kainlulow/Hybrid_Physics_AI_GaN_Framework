import os
import argparse
import numpy as np
import pandas as pd
import re


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reshape Dataset.csv into long format and split per-sample IV CSVs")
    parser.add_argument("--input-csv", default="Data_Extraction/Dataset.csv", help="Path to input Dataset.csv")
    parser.add_argument("--output-csv", default="Data_Extraction/extracted_Id_Vgs_long.csv", help="Path to write the long-format master CSV")
    parser.add_argument("--iv-folder", default="Data_Extraction/IV", help="Folder to write per-sample IV CSVs")
    parser.add_argument("--feature-cols", default="sde_xAl,sde_xIn,sde_tRecess,sde_AlN1,sde_AlN2", help="Comma-separated list of feature columns to keep")
    parser.add_argument("--vgs-start", type=float, default=5, help="Vgs start value")
    parser.add_argument("--vgs-stop", type=float, default=-10, help="Vgs stop value")
    parser.add_argument("--vgs-points", type=int, default=300, help="Number of Vgs points (matches I0..I{N-1})")
    parser.add_argument("--vds-const", type=float, default=-5, help="Constant drain voltage to annotate in outputs")
    return parser


def run(args: argparse.Namespace) -> tuple[str, str]:
    INPUT_CSV = args.input_csv
    OUTPUT_CSV = args.output_csv
    IV_FOLDER = args.iv_folder
    FEATURE_COLS = [c for c in (args.feature_cols.split(",") if args.feature_cols else []) if c]
    VGS_START, VGS_STOP, VGS_POINTS = float(args.vgs_start), float(args.vgs_stop), int(args.vgs_points)
    VDS_CONST = float(args.vds_const)

    df = pd.read_csv(INPUT_CSV)

    # Validate required columns
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")
    if "Nodes" not in df.columns or "SourceFolder" not in df.columns:
        raise ValueError("The input CSV must contain 'Nodes' and 'SourceFolder' columns.")

    # Find Id columns I0..I{N-1} and sort numerically
    id_cols = sorted(
        [c for c in df.columns if c.startswith("I") and c[1:].isdigit()],
        key=lambda x: int(x[1:])
    )
    if len(id_cols) != VGS_POINTS:
        raise ValueError(
            f"Expected {VGS_POINTS} Id columns (I0..I{VGS_POINTS-1}), but found {len(id_cols)}"
        )

    # Build Vgs vector
    Vgs_vec = np.linspace(VGS_START, VGS_STOP, VGS_POINTS)
    vgs_map = {i: Vgs_vec[i] for i in range(VGS_POINTS)}

    # Reshape to long format
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

    # Final column order (master CSV)
    final = melted[FEATURE_COLS + ["Vds", "Vgs", "Ids"]]
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"Master CSV saved to: {OUTPUT_CSV}")

    # Save individual CSV files
    os.makedirs(IV_FOLDER, exist_ok=True)

    melted_full = melted[["row_id", "Nodes", "SourceFolder"] + FEATURE_COLS + ["Vds", "Vgs", "Ids"]]

    for row_id, group_df in melted_full.groupby("row_id"):
        nodes = group_df["Nodes"].iloc[0]
        src_folder = group_df["SourceFolder"].iloc[0]
        safe_src_folder = re.sub(r'[^A-Za-z0-9_-]', '_', str(src_folder))
        out_file = os.path.join(IV_FOLDER, f"sample_{nodes}_{safe_src_folder}.csv")
        sample_df = group_df[FEATURE_COLS + ["Vds", "Vgs", "Ids"]]
        sample_df.to_csv(out_file, index=False)

    print(f"Saved individual IV CSV files in folder: {IV_FOLDER}")
    return OUTPUT_CSV, IV_FOLDER


if __name__ == "__main__":
    parser = build_arg_parser()
    ns = parser.parse_args()
    run(ns)
