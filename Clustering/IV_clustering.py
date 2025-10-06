import os
import argparse
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import skew, kurtosis


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster IV curves and export summaries/plots")
    parser.add_argument("--data-path", default="../Data_Extraction/Dataset.csv", help="Input CSV path from extraction stage")
    parser.add_argument("--output-dir", default="IV_Clusters", help="Output directory for clustering results")
    parser.add_argument("--mostype", choices=["n", "p"], default="p", help="Device type for default V range")
    parser.add_argument("--v0", type=float, default=None, help="Start of Vgs sweep")
    parser.add_argument("--v1", type=float, default=None, help="End of Vgs sweep")
    parser.add_argument("--points", type=int, default=300, help="Number of Vgs points")
    parser.add_argument("--min-k", type=int, default=6, help="Minimum k for KMeans search (inclusive)")
    parser.add_argument("--max-k", type=int, default=30, help="Maximum k for KMeans search (exclusive)")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for KMeans")
    parser.add_argument("--pca-threshold", type=float, default=0.95, help="Cumulative variance threshold for PCA component selection")
    return parser


def _default_v_range(mostype: str):
    if mostype.lower() == 'n':
        return (-1.0, 1.0)
    return (7.0, -10.0)


def slugify_filename(name: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_\-]+", "_", str(name)).strip("_")
    return safe or "metric"


def detect_metric_columns(df: pd.DataFrame, cluster_col: str) -> list:
    exclude = {cluster_col.lower(), "sourcefolder", "source_folder", "vdd", "vov_on", "vov_off", "folder"}
    metric_cols = []
    for col in df.columns:
        if col.lower() in exclude or col == cluster_col:
            continue
        ser = pd.to_numeric(df[col], errors="coerce")
        if ser.notna().sum() > 0:
            metric_cols.append(col)
    return metric_cols


def make_wide_by_cluster(df: pd.DataFrame, metric: str, cluster_col: str) -> pd.DataFrame:
    values = pd.to_numeric(df[metric], errors="coerce")
    clusters = df[cluster_col]
    try:
        uniq = sorted(pd.unique(clusters.astype(float)))
    except Exception:
        uniq = sorted(pd.unique(clusters.astype(str)), key=lambda x: str(x))

    data = {}
    max_len = 0
    for c in uniq:
        mask = clusters == c
        vals = values[mask].dropna().tolist()
        max_len = max(max_len, len(vals))
        try:
            header = f"Cluster {int(float(c))}"
        except Exception:
            header = f"Cluster {c}"
        data[header] = vals

    for k in list(data.keys()):
        if len(data[k]) < max_len:
            data[k] = data[k] + [""] * (max_len - len(data[k]))

    return pd.DataFrame(data)


def run(args: argparse.Namespace) -> str:
    data_path = args.data_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Clean up previous plots
    for f in glob.glob(os.path.join(output_dir, "*.png")):
        os.remove(f)

    # Transistor sweep setup
    if args.v0 is not None and args.v1 is not None:
        V0, V1 = float(args.v0), float(args.v1)
    else:
        V0, V1 = _default_v_range(args.mostype)
    totalPt = int(args.points)
    Vg = np.linspace(V0, V1, totalPt)

    # Load and clean data
    df = pd.read_csv(data_path, na_values=['NA', 'N/A', 'nan', ''])

    # Collect IV columns I0..I300 (numeric sort)
    _iv_pat = re.compile(r'^I(\d+)$')
    iv_cols_with_idx = []
    for col in df.columns:
        m = _iv_pat.match(col)
        if m:
            k = int(m.group(1))
            if 0 <= k <= 300:
                iv_cols_with_idx.append((k, col))
    iv_columns = [col for _, col in sorted(iv_cols_with_idx)]
    if not iv_columns:
        raise ValueError("No IV columns found matching pattern I0..I300 in the input CSV.")

    required_metrics = ['Ioff_fixed0', 'Ion_fixedV', 'SS', 'Vth']
    missing = [c for c in required_metrics if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required metric columns in CSV: {missing}")

    df = df.dropna(subset=iv_columns + required_metrics).reset_index(drop=True)

    # Build IV matrix
    Id_matrix = df[iv_columns].values

    # Step 1: Log Transform
    log_Id_matrix = np.log10(np.abs(Id_matrix) + 1e-16)

    # Step 2: Normalize for Shape
    log_Id_norm = (log_Id_matrix - np.mean(log_Id_matrix, axis=1, keepdims=True)) / (
        np.std(log_Id_matrix, axis=1, keepdims=True) + 1e-10)

    # Step 3: PCA - Auto Component Selection with Plot
    explained_variance_threshold = float(args.pca_threshold)
    pca_temp = PCA()
    pca_temp.fit(log_Id_norm)
    cum_var = np.cumsum(pca_temp.explained_variance_ratio_)
    optimal_pca_components = np.argmax(cum_var >= explained_variance_threshold) + 1

    # Plot PCA cumulative variance
    plt.figure(figsize=(7, 4))
    plt.plot(cum_var, marker='o', label='Cumulative Variance')
    plt.axhline(explained_variance_threshold, color='red', linestyle='--', label='Threshold')
    plt.axvline(optimal_pca_components - 1, color='green', linestyle='--', label=f'Chosen: {optimal_pca_components}')
    plt.title("PCA Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_component_selection.png"))
    plt.close()

    pca_model = PCA(n_components=optimal_pca_components)
    shape_features = pca_model.fit_transform(log_Id_norm)

    # Step 4: Magnitude-Based Features
    METRIC_WEIGHTS = {'Ioff_fixed0': 1.0, 'Ion_fixedV': 1.0, 'SS': 5.0, 'Vth': 1.0}

    def extract_magnitude_features(Vg_vec, Id_vec):
        Id_abs = np.abs(Id_vec)
        Ion = np.max(Id_abs)
        Ioff = np.min(Id_abs)
        on_off_ratio = Ion / (Ioff + 1e-20)
        auc = np.trapz(Id_abs, Vg_vec)
        return [
            np.mean(Id_abs),
            np.log10(np.abs(Id_abs[-1]) + 1e-16),
            skew(Id_abs),
            kurtosis(Id_abs),
            auc,
            # on_off_ratio  # optional
        ]

    magnitude_features = np.array([extract_magnitude_features(Vg, Id) for Id in Id_matrix])

    # Step 4.1: Append Device Metrics
    metric_data = []
    for metric in ['Ioff_fixed0', 'Ion_fixedV']:
        metric_data.append(np.log10(np.abs(df[metric].values) + 1e-16) * METRIC_WEIGHTS[metric])
    for metric in ['SS', 'Vth']:
        metric_data.append(df[metric].values * METRIC_WEIGHTS[metric])
    metric_features = np.column_stack(metric_data)

    # Step 5: Combine + Standardize
    combined_features = np.hstack((shape_features, magnitude_features, metric_features))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(combined_features)

    # Step 6: Optimal Cluster Search
    sil_scores = []
    cluster_range = range(int(args.min_k), int(args.max_k))
    best_k = None
    best_score = -1
    best_model = None
    best_labels = None

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=int(args.random_state))
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        sil_scores.append(score)
        if score > best_score:
            best_k = k
            best_model = kmeans
            best_labels = labels
            best_score = score

    # Final model
    kmeans = best_model
    labels = best_labels
    df['Cluster'] = labels
    outfile = os.path.join(output_dir, "Clustered_IV_Data.csv")
    df.to_csv(outfile, index=False)

    # Step 7: Plot Silhouette Scores
    plt.figure(figsize=(6, 4))
    plt.plot(list(cluster_range), sil_scores, marker='o')
    plt.title("Silhouette Score vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "silhouette_vs_k.png"))
    plt.close()

    # Step 8: PCA Variance Plot
    plt.figure(figsize=(6, 4))
    plt.plot(cum_var, marker='o')
    plt.axhline(float(args.pca_threshold), linestyle='--', color='red')
    plt.title("PCA Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_variance.png"))
    plt.close()

    # Step 9: Plot Cluster-wise IVs
    plot_dir = os.path.join(output_dir, "Cluster_IV_Plots1")
    os.makedirs(plot_dir, exist_ok=True)
    for f in glob.glob(os.path.join(plot_dir, "*")):
        os.remove(f)

    colors = plt.cm.tab10
    unique_clusters = np.unique(labels)

    for cluster_id in unique_clusters:
        fig, axes = plt.subplots(1, 2, figsize=(14, 14))
        cluster_indices = np.where(labels == cluster_id)[0]
        for idx in cluster_indices:
            axes[0].plot(Vg, np.abs(Id_matrix[idx]), alpha=0.4)
            axes[1].semilogy(Vg, (np.abs(Id_matrix[idx]) + 1e-16), alpha=0.4)
        axes[0].set_title(f"Cluster {cluster_id} - Linear")
        axes[0].set_xlabel("Vg (V)")
        axes[0].set_ylabel("Id (A)")
        axes[0].grid(True)
        axes[1].set_title(f"Cluster {cluster_id} - Log10(|Id|)")
        axes[1].set_xlabel("Vg (V)")
        axes[1].set_ylabel("log10(|Id|)")
        axes[1].grid(True)
        plt.suptitle(f"Cluster {cluster_id} - IV Curves")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(plot_dir, f"Cluster_{cluster_id}_IVs_dual.png"))
        plt.close()

        # Save each cluster's abs(Id)-Vg as CSV
        cluster_id_matrix = np.abs(Id_matrix[cluster_indices]).T
        cluster_iv_df = pd.DataFrame(cluster_id_matrix, index=Vg)
        cluster_iv_df.reset_index(inplace=True)
        cluster_iv_df.columns = ['Vg'] + [f'Id_{i}' for i in range(cluster_id_matrix.shape[1])]
        cluster_iv_df.to_csv(os.path.join(output_dir, f"Cluster_{cluster_id}_IV.csv"), index=False)

    # Step 10: Plot All Curves (All Clusters)
    fig, axes = plt.subplots(1, 2, figsize=(14, 14))
    for i in range(len(Id_matrix)):
        axes[0].plot(Vg, abs(Id_matrix[i]), color=colors(labels[i] % 10), alpha=0.4)
        axes[1].semilogy(Vg, (np.abs(Id_matrix[i]) + 1e-16), color=colors(labels[i] % 10), alpha=0.4)
    axes[0].set_title("All Clusters - Linear")
    axes[0].set_xlabel("Vg (V)")
    axes[0].set_ylabel("Id (A)")
    axes[0].grid(True)
    axes[1].set_title("All Clusters - Log10(|Id|)")
    axes[1].set_xlabel("Vg (V)")
    axes[1].set_ylabel("log10(|Id|)")
    axes[1].grid(True)
    plt.suptitle("All IV Curves Colored by Cluster")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(plot_dir, "All_Clusters_IVs_dual.png"))
    plt.close()

    # Step 11: Cluster Metrics Summary
    vth_valid = df[df['Vth'] < 100]
    vth_invalid = df[df['Vth'] >= 100]
    metrics_cols = ['Ioff_fixed0', 'Ion_fixedV', 'SS']
    summary_stats = df.groupby('Cluster')[metrics_cols].agg(['mean', 'std'])
    vth_stats = vth_valid.groupby('Cluster')['Vth'].agg(['mean', 'std'])
    vth_invalid_counts = vth_invalid.groupby('Cluster')['Vth'].count()
    total_counts = df.groupby('Cluster')['Vth'].count()
    vth_ratio = (vth_invalid_counts / total_counts).fillna(0)
    summary_stats[('Vth', 'mean')] = vth_stats['mean']
    summary_stats[('Vth', 'std')] = vth_stats['std']
    summary_stats[('Vth', '>=100 (#/total)')] = vth_ratio
    summary_stats[('Total Samples', '')] = total_counts
    summary_stats.columns = ['_'.join(col).strip() for col in summary_stats.columns]
    summary_stats.reset_index(inplace=True)
    summary_stats.to_csv(os.path.join(output_dir, 'Cluster_Metrics_Summary.csv'), index=False)

    # Metric CSVs by cluster
    input_file = outfile
    output_folder = os.path.join(output_dir, "cluster_metric_exports")
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    df2 = pd.read_csv(input_file)
    cluster_col = "Cluster"
    metrics = ["Vth", "Ion_eq", "Ion_fixedV", "Ioff_eq", "Ioff_fixed0", "ON_OFF_eq", "ON_OFF_fixedV", "SS"]
    if cluster_col not in df2.columns:
        matches = [c for c in df2.columns if c.lower() == cluster_col.lower()]
        if len(matches) == 1:
            cluster_col = matches[0]
        else:
            raise ValueError(f"Cluster column '{cluster_col}' not found. Available: {list(df2.columns)}")
    if not metrics:
        metrics = detect_metric_columns(df2, cluster_col)
    for metric in metrics:
        if metric not in df2.columns:
            print(f"[WARN] Skipping metric not found: {metric}")
            continue
        wide = make_wide_by_cluster(df2, metric, cluster_col)
        fname = f"{slugify_filename(metric)}_by_cluster.csv"
        out_path = Path(output_folder) / fname
        wide.to_csv(out_path, index=False)
        print(f"[OK] Wrote {metric!r} → {out_path}")

    print("\nDone.")
    return outfile


if __name__ == "__main__":
    parser = build_arg_parser()
    ns = parser.parse_args()
    run(ns)
