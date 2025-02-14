import os
import csv
import warnings
import numpy as np
import pandas as pd
import permEmbed
import ordParNet_v1

warnings.filterwarnings('ignore')

def stat_MLP(output_csv, test_mode=False):
    """
    Processes EEG data from different folders, computes graph-based and permutation-based statistics, 
    and saves results to a CSV file.

    Arguments:
        output_csv: str
            Name of the output CSV file.
        test_mode: bool, optional
            If True, only processes one file per dataset folder (default: False).
    """
    dataset_folders = {
        "A": r"Epilepsy\Set A\A",
        "B": r"Epilepsy\Set B\B",
        "C": r"Epilepsy\Set C\C",
        "D": r"Epilepsy\Set D\D",
        "E": r"Epilepsy\Set E\E"
    }

    fieldnames = [
        'file', 'label', 'dim', 'lag', 'n', 'm', 'AD', 'WASPL', 'WGE', 'ACC', 'AFC', 'T',
        'WCC', 'K_in_min', 'Fit_in', 'Param1_in', 'Param2_in', 'K_out_min', 'Fit_out', 
        'Param1_out', 'Param2_out', 'H', 'MPR', 'EMD'
    ]

    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for label, folder_path in dataset_folders.items():
            if os.path.exists(folder_path):  
                file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(".txt")]

                for file in file_list[:5] if test_mode else file_list:  # Process only 1 file if test_mode=True
                    file_name = os.path.splitext(file)[0]  
                    df = pd.read_csv(os.path.join(folder_path, file), delimiter="\t", header=None)
                    dat = df.iloc[::].to_numpy()
                    
                    # lag = permEmbed.optimal_lag(dat, lrange=50, step=1)
                    lag = permEmbed.mutual_info_lag(dat, lrange=30, step=1)
                    dim = permEmbed.optimal_dim(dat, lag=lag, drange=10)
                    G, series, miss_perm = permEmbed.OPN1(dat, dim=dim, lag=lag)

                    # Graph measurements
                    n = G.number_of_nodes()
                    m = G.number_of_edges()
                    AD = round(2 * m / n, 3)

                    WASPL, WGE, WCC = ordParNet_v1.compute_waspl_wge_wcc(G)
                    ACC, AFC, T = ordParNet_v1.compute_clustering(G)
                    fit_in_result, fit_out_result = ordParNet_v1.deg_dis(G)

                    # Permutation metrics
                    H, MPR, EMD = ordParNet_v1.compute_pattern_metrics(series=series, miss_perm=miss_perm, dim=dim)

                    writer.writerow({
                        'file': file_name, 'label': label, 'dim': dim, 'lag': lag, 
                        'n': n, 'm': m, 'AD': AD, 'WASPL': WASPL, 'WGE': WGE,
                        'ACC': ACC, 'AFC': AFC, 'T': T, 'WCC': WCC,
                        'K_in_min': fit_in_result[0], 'Fit_in': fit_in_result[1], 
                        'Param1_in': np.round(fit_in_result[2][0][0], 2), 
                        'Param2_in': np.round(fit_in_result[2][0][1], 2) if len(fit_in_result[2][0]) > 1 else 0, 
                        'K_out_min': fit_out_result[0], 'Fit_out': fit_out_result[1], 
                        'Param1_out': np.round(fit_out_result[2][0][0], 2), 
                        'Param2_out': np.round(fit_out_result[2][0][1], 2) if len(fit_out_result[2][0]) > 1 else 0, 
                        'H': H, 'MPR': MPR, 'EMD': EMD
                    })

                    print(f"Processed: {file_name}, Label: {label}")

output_csv = "MI_stat.csv"
stat_MLP(output_csv, test_mode=False)  # Set test_mode=True to process only 1 file per folder
