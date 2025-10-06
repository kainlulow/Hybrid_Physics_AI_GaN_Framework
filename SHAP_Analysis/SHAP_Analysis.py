"""
SHAP analysis using INPUT FEATURE ARRAYS:
- Input features are taken EXCLUSIVELY from arrays of column names you provide.
- The script matches those names against the CSV columns, preserves your order,
  warns on missing names, and proceeds with the intersection.
- Targets (outputs) are auto-detected by name aliases, or you can force them.

Author: (you)
"""

import os
import re
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ============ Config ============
FILE_PATH = '../Data_Extraction/Dataset.csv' 
OUT_DIR   = "SHAP_FromArrays"

# If you want to force targets, list them here (must exist in CSV). If None, auto-detect by aliases.
FORCED_TARGETS = None  # e.g., ["Ion_eq", "Ion_fixedV", "Ioff_eq", "Ioff_fixed0", "SS", "Vth"]

# Model / split
TEST_SIZE     = 0.2
RANDOM_STATE  = 42
N_ESTIMATORS  = 120
TOP_K_PLOT    = 20

# ============ INPUT FEATURE ARRAYS ============
INPUT_FEATURE_ARRAYS = [['sde_xAl'],	['sde_xIn'], ['sde_tRecess'], ['sde_AlN1'],	['sde_AlN2']]

# ============ Target name aliases ============
TARGET_NAME_ALIASES = {
    "Ion_eq": "Ion_eq", "Ion_fixedV": "Ion_fixedV",
    "Ioff_eq": "Ioff_eq", "Ioff_fixed0": "Ioff_fixed0",
    "ON_OFF_eq": "ON_OFF_eq", "ON_OFF_fixedV": "ON_OFF_fixedV",
    "SS": "SS", "Vth": "Vth",
}

# Columns not to use as inputs (won’t matter much since we now select by arrays)
META_COLS = {"SourceFolder", "Nodes", "sim_index", "__cluster__", "cluster", "device_id", "run_id", "seed"}

# ============ Transforms ============
TARGET_TRANSFORM_POLICY = {
    "Ion_eq":       "signed_log10p",
    "Ion_fixedV":   "signed_log10p",
    "Ioff_eq":      "log10p",
    "Ioff_fixed0":  "log10p",
    "SS":           None,
    "Vth":          None,
    "Ron":          "log10p",
    "Ion":          "signed_log10p",
    "Ioff":         "log10p",
}
AUTO_TRANSFORM_FEATURES = True
RANGE_RATIO_THRESHOLD   = 1e3  # feature transform trigger (based on 99th/1st pct ratio)

# Make output directory
os.makedirs(OUT_DIR, exist_ok=True)

# Numpy <-> shap compat
if not hasattr(np, "bool"):
    np.bool = np.bool_

# ============ Helpers ============
def log10p_pos(x):  # x >= 0
    return np.log10(np.asarray(x, dtype=float) + 1.0)

def inv_log10p_pos(z):
    return np.power(10.0, np.asarray(z, dtype=float)) - 1.0

def signed_log10p(x):  # x can be +/-/0
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log10(np.abs(x) + 1.0)

def inv_signed_log10p(z):
    z = np.asarray(z, dtype=float)
    return np.sign(z) * (np.power(10.0, np.abs(z)) - 1.0)

TRANSFORMS = {
    None:            (lambda v: v, lambda v: v, "none"),
    "log10p":        (log10p_pos, inv_log10p_pos, "log10(1+x)"),
    "signed_log10p": (signed_log10p, inv_signed_log10p, "sign(x)*log10(1+|x|)"),
}

def choose_feature_transform(series, range_ratio_threshold=RANGE_RATIO_THRESHOLD):
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    if s.empty:
        return None
    lo = np.nanpercentile(np.abs(s), 1)
    hi = np.nanpercentile(np.abs(s), 99)
    if lo <= 0:
        nonzero = np.abs(s[s != 0])
        if nonzero.size == 0:
            return None
        lo = np.nanpercentile(nonzero, 1)
    ratio = hi / max(lo, 1e-12)
    if ratio < range_ratio_threshold:
        return None
    is_nonneg = (s.min() >= 0)
    return "log10p" if is_nonneg else "signed_log10p"

def compute_shap_with_fallback(model, X_train_proc, X_proc, max_bg=1024, random_state=RANDOM_STATE):
    # Background for interventional
    if len(X_train_proc) > max_bg:
        bg = shap.utils.sample(X_train_proc, max_bg, random_state=random_state)
    else:
        bg = X_train_proc
    try:
        expl = shap.TreeExplainer(
            model, data=bg, feature_perturbation="interventional", model_output="raw"
        )
        sv = expl.shap_values(X_proc, check_additivity=False)
        ev = expl.expected_value
        algo = "interventional (check_additivity=False)"
    except Exception as e:
        print(f"[WARN] Interventional SHAP failed: {e}\nFalling back to tree_path_dependent.")
        expl = shap.TreeExplainer(
            model, feature_perturbation="tree_path_dependent", model_output="raw"
        )
        sv = expl.shap_values(X_proc)
        ev = expl.expected_value
        algo = "tree_path_dependent"
    if isinstance(sv, list):
        sv_arr = np.asarray(sv[0])
    else:
        sv_arr = np.asarray(sv)
    if sv_arr.ndim == 3:
        sv_arr = sv_arr[:, 0, :]
    elif sv_arr.ndim == 1:
        sv_arr = sv_arr.reshape(-1, 1)
    pred = model.predict(X_proc)
    ev_arr = np.array(ev)
    if ev_arr.ndim == 0:
        ev_arr = np.repeat(ev_arr, len(X_proc))
    add_err = (sv_arr.sum(axis=1) + ev_arr) - pred
    print(f"[SHAP] algo={algo} | additivity max|err|={np.max(np.abs(add_err)):.6g} "
          f"(mean|err|={np.mean(np.abs(add_err)):.6g})")
    return sv_arr, ev, algo

def posneg_stats_with_feature_means(shap_col, x_col):
    s = pd.Series(shap_col)
    x = pd.Series(x_col).reindex(s.index)
    pos_mask = s > 0
    neg_mask = s < 0
    pos = s[pos_mask]; neg = s[neg_mask]
    x_pos = x[pos_mask]; x_neg = x[neg_mask]
    return {
        "shap_mean_abs": s.abs().mean(),
        "shap_mean_pos": pos.mean() if len(pos) else np.nan,
        "shap_mean_neg": neg.mean() if len(neg) else np.nan,
        "count_pos": int(pos_mask.sum()),
        "count_neg": int(neg_mask.sum()),
        "x_mean_when_shap_pos": x_pos.mean() if len(x_pos) else np.nan,
        "x_mean_when_shap_neg": x_neg.mean() if len(x_neg) else np.nan
    }

def plot_posneg_with_xmeans(df_stats, tgt, save_dir, top_k=None):
    d = df_stats.copy().sort_values("shap_mean_abs", ascending=False)
    if top_k is not None:
        d = d.head(top_k)
    y = d["feature"]
    pos_vals = d["shap_mean_pos"].fillna(0.0).values
    neg_vals = d["shap_mean_neg"].fillna(0.0).values
    mu_pos = d["x_mean_when_shap_pos"].values
    mu_neg = d["x_mean_when_shap_neg"].values
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(y))))
    ax.barh(y, neg_vals, label="Mean SHAP (negative)")
    ax.barh(y, pos_vals, label="Mean SHAP (positive)")
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.invert_yaxis()
    maxabs = np.nanmax(np.abs(np.r_[pos_vals, neg_vals])) if len(y) else 0
    scale_pad = 0.02 * maxabs if maxabs > 0 else 0.02
    for i in range(len(y)):
        ax.text(min(neg_vals[i], 0) - scale_pad, i, f"μ⁻={mu_neg[i]:.3g}",
                va="center", ha="right", fontsize=9)
        ax.text(max(pos_vals[i], 0) + scale_pad, i, f"μ⁺={mu_pos[i]:.3g}",
                va="center", ha="left", fontsize=9)
    ax.set_xlabel("Signed SHAP mean (split by sign)")
    ax.set_title(f"{tgt} — SHAP means with conditional feature means (μ⁺, μ⁻)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{tgt}_posneg_with_mu.png"), dpi=300)
    plt.close()

# ============ Column detection ============
def detect_targets_by_name(df, forced_targets=None):
    if forced_targets:
        missing = [t for t in forced_targets if t not in df.columns]
        if missing:
            raise KeyError(f"Forced targets not found in CSV: {missing}")
        return forced_targets, {t: t for t in forced_targets}
    present = {}
    for col in df.columns:
        if col in TARGET_NAME_ALIASES:
            canonical = TARGET_NAME_ALIASES[col]
            present[canonical] = col
    if not present:
        raise ValueError("No target/metric columns found by name. "
                         "Add to TARGET_NAME_ALIASES or set FORCED_TARGETS.")
    preferred_order = ["Ion_eq", "Ion_fixedV", "Ioff_eq", "Ioff_fixed0", "SS", "Vth", "Ron", "Ion", "Ioff"]
    ordered = [t for t in preferred_order if t in present] + \
              [t for t in present if t not in preferred_order]
    return ordered, present  # canonical list, canonical->actual map

def flatten_unique(seq_of_seqs):
    seen = set()
    out = []
    for arr in seq_of_seqs:
        for name in arr:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out

def select_inputs_from_arrays(df, input_feature_arrays, strict=False):
    """
    Match input feature names (provided by arrays) to df.columns.
    - Preserves your array order.
    - Drops duplicates.
    - Warns (or errors if strict=True) on missing names.
    """
    requested = flatten_unique(input_feature_arrays)
    if not requested:
        raise ValueError("No input feature names provided in INPUT_FEATURE_ARRAYS.")
    available = set(df.columns)

    feature_cols = [c for c in requested if c in available]
    missing      = [c for c in requested if c not in available]

    # Remove accidental targets/meta if present
    feature_cols = [c for c in feature_cols if c not in META_COLS and c not in TARGET_NAME_ALIASES]

    if missing:
        msg = f"[WARN] Missing input features not found in CSV (will be ignored): {missing}"
        if strict:
            raise KeyError(msg)
        else:
            print(msg)

    if not feature_cols:
        raise ValueError("After matching, no input features remain. "
                         "Check your arrays and CSV column names.")
    return feature_cols

# ============ Main ============
def main():
    df = pd.read_csv(FILE_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Cluster handling (optional)
    cluster_col = "cluster" if "cluster" in df.columns else None
    if cluster_col is None:
        df["__cluster__"] = "ALL"
        cluster_col = "__cluster__"

    # Targets
    targets, tgt_name_map = detect_targets_by_name(df, FORCED_TARGETS)
    print(f"[INFO] Targets (canonical → actual col):")
    for t in targets:
        print(f"  - {t:12s} -> {tgt_name_map[t]}")

    # Inputs from arrays (exact name matching)
    feature_cols = select_inputs_from_arrays(df, INPUT_FEATURE_ARRAYS, strict=False)
    print(f"[INFO] Using {len(feature_cols)} input features from arrays:\n  {feature_cols}")

    # Decide feature transforms (auto)
    feature_transform_map = {}
    if AUTO_TRANSFORM_FEATURES:
        for feat in feature_cols:
            tr = choose_feature_transform(df[feat])
            feature_transform_map[feat] = tr  # None / 'log10p' / 'signed_log10p'
    print("[INFO] Feature transforms chosen (None/log10p/signed_log10p):")
    for k in feature_cols:
        print(f"  - {k}: {feature_transform_map.get(k)}")

    all_results = []
    metrics_rows = []

    for tgt in targets:
        col_y = tgt_name_map[tgt]
        if col_y not in df.columns:
            print(f"[Skip] {tgt} not found.")
            continue

        # Prepare X (processed) + originals
        X_orig = df[feature_cols].copy()
        X_proc = X_orig.copy()
        for feat in feature_cols:
            tr_key = feature_transform_map.get(feat, None)
            fwd, inv, _ = TRANSFORMS[tr_key]
            X_proc[feat] = fwd(X_proc[feat])

        # Prepare y (transform by policy; fallback to auto if None)
        y_orig = df[col_y].copy()
        tr_key_y = TARGET_TRANSFORM_POLICY.get(tgt, None)
        if tr_key_y is None:
            tr_key_y = choose_feature_transform(y_orig)
        fwd_y, inv_y, desc_y = TRANSFORMS[tr_key_y]
        y = pd.Series(fwd_y(y_orig), index=y_orig.index)

        # Drop non-finite
        mask = np.isfinite(X_proc.values).all(axis=1) & np.isfinite(y.values)
        X_proc, X_orig_masked = X_proc.loc[mask], X_orig.loc[mask]
        y, y_orig_masked = y.loc[mask], y_orig.loc[mask]

        # Train/test
        X_train_proc, X_test_proc, y_train, y_test = train_test_split(
            X_proc, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        y_test_orig = y_orig_masked.loc[y_test.index]

        # Model
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        )
        model.fit(X_train_proc, y_train)

        # Metrics in original units
        y_pred_trans = model.predict(X_test_proc)
        y_pred_orig  = inv_y(y_pred_trans)
        r2  = r2_score(y_test_orig, y_pred_orig)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)
        metrics_rows.append({"target": tgt, "R2_orig": r2, "MAE_orig": mae,
                             "y_transform": desc_y})
        print(f"[METRICS] {tgt}: (orig units) R2={r2:.4f}, MAE={mae:.3g} | y-transform={desc_y}")

        # SHAP
        shap_values_arr, expected_value, algo_used = compute_shap_with_fallback(
            model, X_train_proc, X_proc
        )
        if shap_values_arr.shape[1] != len(feature_cols):
            raise RuntimeError(f"SHAP feature mismatch: {shap_values_arr.shape[1]} vs {len(feature_cols)}")
        shap_df = pd.DataFrame(shap_values_arr, columns=feature_cols, index=X_proc.index)

        # Stats
        rows = []
        for feat in feature_cols:
            stats = posneg_stats_with_feature_means(shap_df[feat].values, X_proc[feat].values)
            stats.update({"target": tgt, "feature": feat})
            rows.append(stats)
        res_df = (pd.DataFrame(rows)[[
            "target","feature","shap_mean_abs","shap_mean_pos","shap_mean_neg",
            "count_pos","count_neg","x_mean_when_shap_pos","x_mean_when_shap_neg"
        ]].sort_values("shap_mean_abs", ascending=False))

        # Save CSV
        csv_path = os.path.join(OUT_DIR, f"{tgt}_SHAP_posneg_with_mu.csv")
        res_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"{tgt} SHAP stats → {csv_path}")

        # Plot
        plot_posneg_with_xmeans(res_df, tgt, OUT_DIR, top_k=min(TOP_K_PLOT, len(feature_cols)))

        all_results.append(res_df)

        # Per-simulation export (by cluster + combined)
        per_sim_dir = os.path.join(OUT_DIR, "PerSim_ByCluster")
        os.makedirs(per_sim_dir, exist_ok=True)
        y_pred_trans_full = model.predict(X_proc)
        y_pred_orig_full  = inv_y(y_pred_trans_full)

        per_sim = pd.concat(
            [
                X_proc.reset_index(drop=False).rename(columns={"index": "sim_index"}),
                shap_df.add_prefix("SHAP_").reset_index(drop=True),
            ], axis=1
        )
        cluster_col = "cluster" if "cluster" in df.columns else "__cluster__"
        per_sim[cluster_col] = df.loc[X_proc.index, cluster_col].values
        per_sim["target"] = tgt
        per_sim["y_true_trans"] = y.loc[X_proc.index].values
        per_sim["y_pred_trans_model_on_X"] = y_pred_trans_full
        per_sim["y_true_orig"] = y_orig_masked.loc[X_proc.index].values
        per_sim["y_pred_orig_model_on_X"] = y_pred_orig_full
        for feat in feature_cols:
            per_sim[f"ORIG_{feat}"] = X_orig_masked.loc[X_proc.index, feat].values

        id_cols = ["target", cluster_col, "sim_index",
                   "y_true_orig", "y_pred_orig_model_on_X",
                   "y_true_trans", "y_pred_trans_model_on_X"]
        shap_cols = [f"SHAP_{c}" for c in feature_cols]
        orig_feat_cols = [f"ORIG_{c}" for c in feature_cols]
        per_sim = per_sim[id_cols + orig_feat_cols + feature_cols + shap_cols]

        for clus_val, g in per_sim.groupby(cluster_col, dropna=False):
            safe_clus = str(clus_val).replace(os.sep, "_")
            out_path = os.path.join(per_sim_dir, f"{tgt}__cluster_{safe_clus}__per_sim_SHAP.csv")
            g.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"[PerSim] {tgt} cluster={clus_val} → {out_path}")

        combined_path = os.path.join(per_sim_dir, f"{tgt}__ALL_CLUSTERS__per_sim_SHAP.csv")
        per_sim.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"[PerSim] {tgt} combined → {combined_path}")

    # Combined summaries & metrics
    if all_results:
        all_df = pd.concat(all_results, ignore_index=True)
        all_df = all_df.sort_values(["target", "shap_mean_abs"], ascending=[True, False])
        all_df.to_csv(os.path.join(OUT_DIR, "ALL_targets_SHAP_posneg_with_mu.csv"),
                      index=False, encoding="utf-8-sig")
        print("Combined SHAP stats saved.")

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(OUT_DIR, "MODEL_metrics_per_target.csv"),
                      index=False, encoding="utf-8-sig")
    print("Per-target model metrics (in original units) saved.")

if __name__ == "__main__":
    main()
