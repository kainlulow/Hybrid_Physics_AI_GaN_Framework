#!/usr/bin/env python
# coding: utf-8

# Analysis of traps on transistor I-V 

from __future__ import division
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import os
from scipy.interpolate import interp1d
#from scipy import unique
from sklearn.preprocessing import StandardScaler

def normalize_logId_Vg(Id, Vg, Id_min=1e-16):
    """
    Normalize log-scaled drain current and gate voltage.

    Parameters:
    - Id: Array-like of drain current values (can be negative for P-type)
    - Vg: Array-like of gate voltage values
    - Id_min: Minimum absolute Id to avoid log(0) issues (default: 1e-16)

    Returns:
    - norm_log_Id: Normalized log10(|Id|) values
    - norm_Vg: Normalized Vg values
    - mask: Boolean mask for retained points
    """
    Id = np.array(Id)
    Vg = np.array(Vg)

    # Mask for valid values above Id_min
    mask = np.abs(Id) > Id_min
    safe_Id = np.clip(np.abs(Id[mask]), Id_min, None)

    # Compute log10(|Id|) and normalize
    log_Id = np.log10(safe_Id)
    norm_log_Id = StandardScaler().fit_transform(log_Id.reshape(-1, 1)).flatten()

    # Normalize Vg in the same filtered region
    Vg_filtered = Vg[mask]
    norm_Vg = StandardScaler().fit_transform(Vg_filtered.reshape(-1, 1)).flatten()

    return norm_log_Id, norm_Vg, mask

def extract_data(node, V, Voff, Von, Order, folder_path):
    filename = os.path.join(folder_path, f"IdVgsSat_n{int(node)}_des.plt")
    try:
        return Ion_SS_extraction(filename, V, Voff, Von, Order)
    except (FileNotFoundError, PermissionError) as e:
        print(f"Error with file {filename}: {e}, skipping...")
        return None

def run_clustering_and_score(X, feature_min, feature_max, cluster_min, cluster_max):
    feature_cluster_Sil_pca_list, feature_cluster_db_pca_list = [], []
    feature_cluster_Sil_ica_list, feature_cluster_db_ica_list = [], []

    for n in range(feature_min, feature_max):
        # PCA and ICA transformations
        pca = PCA(n_components=n)
        ica = FastICA(n_components=n)
        X_p = pca.fit_transform(X)
        X_i = ica.fit_transform(X)

        # Run clustering and append scores
        score_sil_n, score_db_n = search_clusters_all(X_p, cluster_min, cluster_max)
        feature_cluster_Sil_pca_list.append(score_sil_n.copy())
        feature_cluster_db_pca_list.append(score_db_n.copy())

        score_sil_n, score_db_n = search_clusters_all(X_i, cluster_min, cluster_max)
        feature_cluster_Sil_ica_list.append(score_sil_n.copy())
        feature_cluster_db_ica_list.append(score_db_n.copy())

    return (feature_cluster_Sil_pca_list, feature_cluster_db_pca_list,
            feature_cluster_Sil_ica_list, feature_cluster_db_ica_list)

def save_cluster_plots(df, clusters, V, totalPt, output_path):
    """
    Generate and save individual plots for each class and a combined plot with color differentiation.
    
    Parameters:
    - df: DataFrame containing the data.
    - clusters: List of class labels.
    - V: Gate voltage values.
    - totalPt: Number of points in the data.
    - output_path: Directory to save the plots.
    """
    # Calculate global min and max for the linear scale (Id)
    all_Id_values = []
    for idx in df.index:
        all_Id_values.extend(df.loc[idx, [f"I{i}" for i in range(totalPt)]].values)
    #global_min = min(abs(all_Id_values))
    #global_max = max(abs(all_Id_values))
    global_min = min(abs(x) for x in all_Id_values)  # Use a generator expression to apply abs()
    global_max = max(abs(x) for x in all_Id_values)  # Same for max()


    # Calculate global min and max for the log scale (filter non-positive values)
    #log_all_Id_values = abs(all_Id_values) #[val for val in all_Id_values if val > 0]
    log_global_min = global_min # min(log_all_Id_values)
    log_global_max = global_max #max(log_all_Id_values)

    # Colors for each class
    class_colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))

    # Generate individual plots for each class
    for cluster_idx, cluster in enumerate(clusters):
        plt.figure(figsize=(12, 8))
        index_list = df[df['Class'] == cluster].index

        for idx in index_list:
            Vg = V
            Id = df.loc[idx, [f"I{i}" for i in range(totalPt)]].values

            # Linear Scale Plot
            plt.subplot(1, 2, 1)
            plt.plot(Vg, abs(Id), color=class_colors[cluster_idx], alpha=0.7)
            plt.xlabel('Gate Voltage, $V_g$ (V)', fontsize=16)
            plt.ylabel('Drain Current, $I_d$ (A)', fontsize=16)
            plt.title(f'Class {cluster} - Linear Scale', fontsize=18)
            plt.ylim(global_min, global_max)
            plt.xlim(min(V), max(V))

            plt.grid(True, linestyle='--', linewidth=0.5)

            # Log Scale Plot
            plt.subplot(1, 2, 2)
            positive_Id = abs(Id) #[val if val > 0 else np.nan for val in Id]
            plt.plot(Vg, positive_Id, color=class_colors[cluster_idx], alpha=0.7)
            plt.yscale('log')
            plt.xlabel('Gate Voltage, $V_g$ (V)', fontsize=16)
            plt.ylabel('Drain Current, $I_d$ (A)', fontsize=16)
            plt.title(f'Class {cluster} - Log Scale', fontsize=18)
            plt.ylim(log_global_min, log_global_max)
            plt.xlim(min(V), max(V))
            plt.grid(True, linestyle='--', linewidth=0.5)

        # Save individual class plot
        filename = os.path.join(output_path, f'class_{cluster}.png')
        plt.tight_layout()
        plt.savefig(filename, format='png', dpi=300)
        plt.close()

    # Generate combined plot for all classes
    plt.figure(figsize=(12, 8))

    for cluster_idx, cluster in enumerate(clusters):
        index_list = df[df['Class'] == cluster].index
        for idx in index_list:
            Vg = V
            Id = df.loc[idx, [f"I{i}" for i in range(totalPt)]].values

            # Linear Scale Plot
            plt.subplot(1, 2, 1)
            plt.plot(Vg, abs(Id), color=class_colors[cluster_idx], alpha=0.7, label=f'Class {cluster}' if idx == index_list[0] else "")
            plt.xlabel('Gate Voltage, $V_g$ (V)', fontsize=16)
            plt.ylabel('Drain Current, $I_d$ (A)', fontsize=16)
            plt.title('Combined - Linear Scale', fontsize=18)
            plt.ylim(global_min, global_max)
            plt.grid(True, linestyle='--', linewidth=0.5)

            # Log Scale Plot
            plt.subplot(1, 2, 2)
            positive_Id = abs(Id) #[val if val > 0 else np.nan for val in Id]
            plt.plot(Vg, positive_Id, color=class_colors[cluster_idx], alpha=0.7, label=f'Class {cluster}' if idx == index_list[0] else "")
            plt.yscale('log')
            plt.xlabel('Gate Voltage, $V_g$ (V)', fontsize=16)
            plt.ylabel('Drain Current, $I_d$ (A)', fontsize=16)
            plt.title('Combined - Log Scale', fontsize=18)
            plt.ylim(log_global_min, log_global_max)
            plt.grid(True, linestyle='--', linewidth=0.5)

    # Add legend for the classes
    plt.subplot(1, 2, 1)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10, title='Classes')

    plt.subplot(1, 2, 2)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best', fontsize=10, title='Classes')

    # Save the combined plot
    filename = os.path.join(output_path, 'combined_clusters_colored.png')
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300)
    plt.close()


def search_clusters_all(X_scaled, cluster_min, cluster_max):
    # Searching through clusters
    km_scores= []
    km_silhouette = []
    db_score = []
    for i in range(cluster_min, cluster_max):
        #print(i)
        km = KMeans(n_clusters=i, random_state=0, n_init='auto').fit(X_scaled)
        preds = km.predict(X_scaled)
        #print("Elbow cluster(s) {}: {}".format(i,km.score(X_scaled)))
        km_scores.append(-km.score(X_scaled))

        silhouette = silhouette_score(X_scaled,preds)
        km_silhouette.append(silhouette)
        #print("Silhouette cluster(s) {}: {}".format(i,silhouette))
    
        db = davies_bouldin_score(X_scaled,preds)
        db_score.append(db)
        #print("Davies Bouldin cluster(s) {}: {}".format(i,db))
    #print()
    #print("Elbow min {} at {}".format(min(km_scores), km_scores.index(min(km_scores))+cluster_min))
    #print("Silhouette max {} at {}".format(max(km_silhouette), km_silhouette.index(max(km_silhouette))+cluster_min))
    #print("Davies-Bouldin min {} at {}".format(min(db_score), db_score.index(min(db_score))+cluster_min))
# #     clustering_score(km_scores, km_silhouette, db_score, cluster_max)
    return km_silhouette, db_score

def clustering_score_analysis(feature_cluster, feature_min, cluster_min, scoretype, descend):
    array = np.array(feature_cluster)
    # Get the indices that would sort the flattened array in descending order
    if descend == 1:
        sorted_indices = np.argsort(array, axis=None)[::-1]
    else:
        sorted_indices = np.argsort(array, axis=None)
    # Select the indices of the top 10 elements
    top_10_indices = sorted_indices[:10]
    # Convert the flat indices to 2D indices
    top_10_2d_indices = np.unravel_index(top_10_indices, array.shape)
    # Get the top 10 elements
    top_10_elements = array[top_10_2d_indices]
    adjusted_indices = (top_10_2d_indices[0] + feature_min, top_10_2d_indices[1] + cluster_min)
    #top_10_elements_and_indices = list(zip(list(zip(*adjusted_indices)), top_10_elements))
    top_10_elements_and_indices = [list(index_pair) + [value] for index_pair, value in zip(zip(*adjusted_indices), top_10_elements)]
    print(f"(# feature, # cluster), {scoretype} Score")

    print('\n'.join(map(str, top_10_elements_and_indices)))
    return top_10_elements_and_indices


def read_gtree(filename):
    simulation_flow = []
    simulation_tree = []

    with open(filename, "r") as f:

        line = f.readline()
        while line:
            if line.split():

                if line.split()[-1] == "flow":
                    line = f.readline()
                    while line:
                        simulation_flow.append([line.split()[0]+"_"+line.split()[1]])
                        line = f.readline()
                        if line.split()[-1] == "variables":
                            break

                if line.split()[-1] == "tree":
                    line = f.readline()
                    while line:
                        if not line.split()[3].split("{")[-1].split("}")[0]:
                            tmp_tree = "-"
                        else:
                            tmp_tree = line.split()[3].split("{")[-1].split("}")[0]
                        simulation_tree.append([line.split()[0],
                                        line.split()[1], line.split()[2], 
                                        tmp_tree])
                        line = f.readline()
                        if not line.split():
                            break

            line = f.readline()

    numcol = len(simulation_flow) - 1

    row_N = [None] * len(simulation_flow) 
    row_V = [None] * len(simulation_flow) 
    Workbench_N = []
    Workbench_V = []
    Variable_name = [elem for sublist in simulation_flow for elem in sublist]

    for i in range(len(simulation_tree)):
        if int(simulation_tree[i][0]) == numcol:
            row_N[int(simulation_tree[i][0])] = simulation_tree[i][1]
            row_V[int(simulation_tree[i][0])] = simulation_tree[i][-1]

            Workbench_N.append(row_N.copy())
            Workbench_V.append(row_V.copy())
        else:
            row_N[int(simulation_tree[i][0])] = simulation_tree[i][1]
            row_V[int(simulation_tree[i][0])] = simulation_tree[i][-1]

    Workbench_N.insert(0, Variable_name)
    Workbench_V.insert(0, Variable_name)

    return Workbench_N, Workbench_V

def read_plt(filename):
    try:
        name_tmp = []
        data_tmp = []
        variable_name = []
        data_result = []

        with open(filename, "r") as f:
            line = f.readline()
            while line:
                if line.split():
                    if line.strip().split()[0] == "datasets":
                        while line:
                            line = f.readline()
                            for value in line.split():
                                if value.strip('"') != "]":
                                    name_tmp.append(value.strip('"'))
                            if line.strip().split()[-1] == "]":
                                break

                    if line.strip().split()[0] == "Data":
                        while line:
                            line = f.readline()
                            for value in line.split():
                                data_tmp.append(value.strip('"'))
                            if line.strip().split()[-1] == "}":
                                break
                line = f.readline()

        # Process the collected names
        for i in range(len(name_tmp)):
            if i == 0:
                variable_name.append(name_tmp[i])
            if (i != 0) & (i % 2 == 0):
                variable_name.append(name_tmp[i - 1] + "_" + name_tmp[i])

        # Process the collected data
        time_tmp = []
        for i in range(len(data_tmp[:-1])):
            time_tmp.append(float(data_tmp[i]))
            if len(time_tmp) == len(variable_name):
                data_result.append(time_tmp)
                time_tmp = []

        # Insert variable names as the first row
        data_result.insert(0, variable_name)
        T = pd.DataFrame(data_result[1:] , columns=data_result[0])
        csv_filename = filename.replace('.plt', '.csv')

        T.to_csv(csv_filename, index=False)
        #T = pd.DataFrame(alldata, columns=dataset)
        #T.to_csv(f'{filename[:-4]}.csv', index=False)

        return data_result

    except FileNotFoundError:
        print(f"Error: The file '{filename}' does not exist.")

        return None  # Return None to indicate the file was not found


def Ion_SS_extraction(filename, V, Voff, Von, Order):
    # Form the name of the CSV file
    filenames = f'{filename[:-4]}.csv'
    
    try:
        # Try to read the CSV file
        IVdata = pd.read_csv(filenames)
        Vg_vec = IVdata['gate_OuterVoltage']
        Id_vec = IVdata['drain_TotalCurrent']
    except FileNotFoundError:
        # If the CSV file doesn't exist, call read_IV_plt
        data_result = read_plt(filename)
        if not data_result:
            I = np.full(len(V), np.nan)
            Vgvec = np.full(len(V), np.nan)
            Idvec = np.full(len(V), np.nan)
            SS = Ioff = Ion = Vgvec = Idvec = np.nan
            return I, SS, Ioff, Ion, Vgvec, Idvec
        else:
            df = pd.DataFrame(data_result[1:], columns=data_result[0])
            Vg_vec = df['gate_OuterVoltage']
            Id_vec = df['drain_TotalCurrent']
    # Remove duplicate values in Vg_vec and corresponding values in Id_vec
    # Extract columns from DataFrame
    _, index = np.unique(Vg_vec, return_index=True)
    Vgvec = Vg_vec[np.sort(index)]
    Idvec = Id_vec[np.sort(index)]
    Idvec.reset_index(drop=True, inplace=True)
    
    # Initialize I
    I = np.zeros(len(V))
    
    # Create interpolation function
    interp_func_I = interp1d(Vgvec, Idvec, kind='linear', fill_value="extrapolate")
    # Create interpolation function
    interp_func_V = interp1d(Idvec, Vgvec, kind='linear', fill_value="extrapolate")
    
    # Interpolate Id values for each value in V
    for i in range(len(V)):
        I[i] = interp_func_I(V[i])
        assert np.isreal(I[i]), f'Imaginary value encountered at index {i}'
    #Voff = -0.45
    #Von = 0.55
    Ioff = interp_func_I(Voff)
    Ion = interp_func_I(Von)
    # Calculate SS
    #order = 1e3
    V1 = interp_func_V(Ioff * Order)
    V0 = interp_func_V(Ioff)
    SS = abs(1000 * (V1 - V0) / np.log10(Order))
    
    return I, SS, Ioff, Ion, Vgvec, Idvec
