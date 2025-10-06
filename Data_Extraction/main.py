import os
import numpy as np
import pandas as pd
import mylibrary as mylib  # updated library

# --- Paths & I/O ---
input_path  = '../Simulation_Data/'
root_folder = os.path.join(input_path, '.')   # contains plt*
output_path = '.'
os.makedirs(output_path, exist_ok=True)

outfile_combined = os.path.join(output_path, 'Dataset.csv')

# --- Device type & sweep setup ---
mostype = 'p'  # 'n' or 'p' (used by metrics layer)
if mostype == 'n':
    V0, V1 = -1.0, 1.5
else:
    V0, V1 = 5.0, -10.0

totalPt = 300
V = np.linspace(V0, V1, totalPt)  # common grid for clustering features

# Metrics configuration
ITH_ABS      = 1e-5
VDD          = -5.0 if mostype.lower() == 'p' else 5.0
VOV_ON_ABS   = 5.0   # equal-overdrive magnitude for Ion
VOV_OFF_ABS  = 2.0   # equal-overdrive magnitude for Ioff (subthreshold)
IOFF_BIAS_V  = 0.0   # fixed-bias Ioff at Vgs = 0 V

def _dedup_vg(Vg_series, Id_series):
    """Remove duplicate Vg entries and return numpy arrays."""
    Vg = np.asarray(Vg_series, dtype=float)
    Id = np.asarray(Id_series, dtype=float)
    _, idx = np.unique(Vg, return_index=True)  # keep first occurrence
    idx = np.sort(idx)
    return Vg[idx], Id[idx]

def _fmt(x, unit=""):
    """Pretty float formatter for console."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NaN" + (f" {unit}" if unit else "")
    return f"{x:.3e}{(' ' + unit) if unit else ''}"

# --- Process one plt subfolder ---
def process_one_folder(folder_path: str, tag: str):
    """
    Process a single subfolder containing gtree.dat and IdVg_*.plt files.
    Returns a DataFrame with per-node metadata, metrics, and I0..I{N-1}.
    """
    gtree_file = os.path.join(folder_path, 'gtree.dat')
    if not os.path.isfile(gtree_file):
        print(f"[WARN] Missing gtree.dat in {folder_path}, skipping.")
        return None

    try:
        Workbench_N, Workbench_V = mylib.read_gtree(gtree_file)
    except Exception as e:
        print(f"[ERROR] read_gtree failed for {folder_path}: {e}")
        return None

    TN = pd.DataFrame(Workbench_N[1:], columns=Workbench_N[0])
    TV = pd.DataFrame(Workbench_V[1:], columns=Workbench_V[0])

    rows_metrics = []
    I_matrix = []
    node_list = []

    print(f"[INFO] Processing folder {tag} ({folder_path}) with {len(TN)} nodes...")

    for idx in range(len(TN)):
        node = TN.at[idx, 'IdVg_gate']
        plt_name = f"IdVg_n{int(node)}_des.plt"
        plt_path = os.path.join(folder_path, plt_name)

        # Read PLT -> DataFrame
        try:
            data_result = mylib.read_plt(plt_path)
            if not data_result:
                print(f"[WARN] Empty/invalid PLT: {plt_path}", flush=True)
                continue
            T = pd.DataFrame(data_result[1:], columns=data_result[0])
            # Save raw per-node CSV (optional)
            #per_node_csv = os.path.join(output_path, f"{tag}_IdVg_n{int(node)}_des.csv")
            #T.to_csv(per_node_csv, index=False)
        except Exception as e:
            print(f"[WARN] read_plt failed: {plt_path} | {e}", flush=True)
            continue

        # Extract columns
        if 'gate_OuterVoltage' not in T.columns or 'drain_TotalCurrent' not in T.columns:
            print(f"[WARN] Missing columns in {plt_path}, skipping.", flush=True)
            continue

        Vg_raw = T['gate_OuterVoltage']
        Id_raw = T['drain_TotalCurrent']
        Vgvec, Idvec = _dedup_vg(Vg_raw, Id_raw)

        # --- Metrics extraction (new) ---
        try:
            m = mylib.extract_metrics_from_curve(
                Vgvec, Idvec,
                mostype=mostype,
                Ith_abs=ITH_ABS,
                Vdd=VDD,
                overdrive_on_abs=VOV_ON_ABS,
                overdrive_off_abs=VOV_OFF_ABS,
                Ioff_bias_V=IOFF_BIAS_V
            )
        except Exception as e:
            print(f"[WARN] metrics extraction failed: {plt_path} | {e}", flush=True)
            continue

        # --- Resample IV on common grid for clustering (use |Id|) ---
        try:
            Igrid = mylib.resample_iv_on_grid(Vgvec, Idvec, V, use_abs=True)
        except Exception as e:
            print(f"[WARN] resample_iv_on_grid failed: {plt_path} | {e}", flush=True)
            continue

        # Save metrics & I(V)
        rows_metrics.append({
            "Vth":          m.get('Vth', np.nan),
            "Ion_eq":       m.get('Ion_eq', np.nan),
            "Ion_fixedV":   m.get('Ion_fixedV', np.nan),
            "Ioff_fixed0":  m.get('Ioff_fixed0', np.nan),
            "Ioff_eq":      m.get('Ioff_eq', np.nan),
            'ON_OFF_eq':    m.get('ON_OFF_eq', np.nan),
            'ON_OFF_fixedV':m.get('ON_OFF_fixedV', np.nan),
            "SS":           m.get('SS_mVdec', np.nan),
            "Vov_on":       m.get('Vov_on', np.nan),
            "Vov_off":      m.get('Vov_off', np.nan),
            "Vdd":          m.get('Vdd', np.nan),
        })
        I_matrix.append(Igrid)
        node_list.append(int(node))

        # ---- REAL-TIME CONSOLE LINE ----
        print(
            f"[{tag}] {plt_name}  "
            f"Vth={_fmt(m.get('Vth'),'V')}, "
            f"Ion(Vdd)={_fmt(m.get('Ion_fixedV'),'A')}, "
            f"Ion(Vov)={_fmt(m.get('Ion_eq'),'A')}, "
            f"Ioff@0V={_fmt(m.get('Ioff_fixed0'),'A')}, "
            f"Ioff(Vov_off)={_fmt(m.get('Ioff_eq'),'A')}, "
            f"ON_OFF_eq={_fmt(m.get('ON_OFF_eq'),'A')}, "
            f"ON_OFF_fixedV={_fmt(m.get('ON_OFF_fixedV'),'A')}, "
            f"SS={_fmt(m.get('SS_mVdec'),'mV/dec')}",
            flush=True
        )

    if not rows_metrics:
        print(f"[WARN] No valid results in {folder_path}.")
        return None

    # Align TV with processed nodes only
    TV = TV.iloc[:len(node_list)].copy()
    TV.loc[:, "Nodes"] = node_list

    # I0..I{N-1} columns
    I_array = np.vstack(I_matrix)
    I_df = pd.DataFrame(I_array, columns=[f"I{i}" for i in range(I_array.shape[1])])

    # Metrics DataFrame
    M_df = pd.DataFrame(rows_metrics)

    # Assemble final table
    out_df = pd.concat([TV.reset_index(drop=True),
                        I_df.reset_index(drop=True),
                        M_df.reset_index(drop=True)], axis=1)
    out_df["SourceFolder"] = tag
    return out_df

# --- Process multiple subfolders: choose which plt* to include ---
all_results = []
for i in range(1, 4):  # adjust as needed
    sub_tag = f"plt{i}"
    sub_folder = os.path.join(root_folder, sub_tag)
    if not os.path.isdir(sub_folder):
        print(f"[WARN] {sub_folder} not found, skipping.")
        continue
    df = process_one_folder(sub_folder, sub_tag)
    if df is not None and not df.empty:
        all_results.append(df)

if not all_results:
    raise RuntimeError("No data collected from any plt* folder. Check inputs.")

# --- Combine & write outputs ---
final_dataset = pd.concat(all_results, ignore_index=True)
final_dataset.to_csv(outfile_combined, index=False, na_rep='NaN')
