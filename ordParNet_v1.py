import os
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import permEmbed
import math
from scipy.stats import entropy
from collections import Counter
import MLE_functions_v2
from scipy.stats import wasserstein_distance
import math
import itertools

warnings.filterwarnings('ignore')

def compute_pattern_metrics(series, miss_perm, dim):
    """
    Computes permutation entropy, missing permutation ratio (MPR), and 
    Earth Mover's Distance (EMD) based on permutation indices.

    Arguments:
        series: list
            List of observed permutations.
        miss_perm: list
            List of missing permutations.
        dim: int
            Embedding dimension.

    Returns:
        pattern_entropy: float
            Shannon entropy of observed permutation frequencies.
        mpr: float
            Missing permutation ratio.
        emd_value: float
            Earth Mover's Distance based on permutation indices.
    """
    # Compute observed permutation frequencies
    pattern_counts = Counter(series)
    total_patterns = len(series)
    pattern_freq = {p: count / total_patterns for p, count in pattern_counts.items()}

    # Compute Shannon entropy
    pattern_entropy = entropy(list(pattern_freq.values()))  

    # Compute Missing Permutation Ratio (MPR)
    total_possible_permutations = math.factorial(dim)
    mpr = len(miss_perm) / total_possible_permutations  

    # Generate all possible permutations lexicographically
    all_perms = [''.join(p) for p in itertools.permutations(''.join(map(str, range(1, dim + 1))))]

    # Align probability distributions for EMD
    observed_freq = [pattern_freq.get(p, 0) for p in all_perms]  # Observed distribution
    ideal_uniform = [1 / total_possible_permutations] * total_possible_permutations  # Uniform distribution
    
    # Compute EMD using properly aligned distributions
    emd_value = wasserstein_distance(range(total_possible_permutations), range(total_possible_permutations), observed_freq, ideal_uniform)

    return pattern_entropy, mpr, emd_value

# Measurement of integration (WASPL, WGE)
def compute_waspl_wge_wcc(G):
    if not nx.is_strongly_connected(G):
        WASPLs, WGEs, WCCs = [], [], []
        
        for C in (G.subgraph(c).copy() for c in nx.strongly_connected_components(G)):
            n = len(C)
            if n > 1 and nx.get_edge_attributes(G, "weight"):  # Check for weights
                wd_ij = [
                    l for u in C for l in nx.single_source_dijkstra_path_length(C, u, weight="weight").values()
                    if l > 0  # Exclude self-distances
                ]
                WASPLs.append(sum(wd_ij) / (n * (n - 1)))
                WGEs.append(sum(1 / d for d in wd_ij) / (n * (n - 1)))
                WCCs.append((n * (n - 1)) / sum(wd_ij))
        WASPL = round(sum(WASPLs) / len(WASPLs), 3) if WASPLs else 0
        WGE = round(sum(WGEs) / len(WGEs), 3) if WGEs else 0
        WCC = round(sum(WCCs) / len(WCCs), 3) if WCCs else 0
        
    else:
        n = len(G)
        wd_ij = [
            l for u in G for l in nx.single_source_dijkstra_path_length(G, u, weight="weight").values()
            if l > 0  # Exclude self-distances
        ]
        WASPL = round(sum(wd_ij) / (n * (n - 1)), 3)
        WGE = round(sum(1 / d for d in wd_ij) / (n * (n - 1)), 3)
        WCC = round((n * (n - 1)) / sum(wd_ij), 3)

    return WASPL, WGE, WCC

# Measurements of segregation 

def prepare_matrices(G):
    """
    Prepare the required matrices for clustering coefficient calculation.
    """
    # Create node to index mapping
    node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}

    # Get adjacency matrix and remove self-loops
    W = nx.adjacency_matrix(G).toarray()
    np.fill_diagonal(W, 0)
    
    # Create binary adjacency matrix
    A = (W != 0).astype(int)
    
    # Calculate required matrix combinations
    W_mat = W.T + W
    A_mat = A.T + A
    W_A_mat = W * A + A * W
    A_2_mat = np.matmul(A, A)
    
    # Normalize W matrix
    max_weight = np.max(W)
    if max_weight > 0:  # Avoid division by zero
        W_nor = W / max_weight
        W_nor_mat = np.power(W_nor, 1/3)
    else:
        W_nor_mat = W
    W_nor_sum = W_nor_mat + W_nor_mat.T
    
    cle_mat = np.matmul(W_mat, np.matmul(A_mat, A_mat))
    fag_mat = np.linalg.matrix_power(W_nor_sum, 3)
    return A_2_mat, W_mat, A_mat, W_A_mat, node_to_idx, cle_mat, fag_mat
    
def compute_clustering(G):
    """
    Compute Clemente and Fagiolo clustering coefficients for each node.
    
    Args:
        G: NetworkX directed graph
        
    Returns:
        tuple: (clemente_clustering, fagiolo_clustering) dictionaries mapping node to clustering coefficient
    """
    # Prepare all required matrices
    A_2_mat, W_mat, A_mat, W_A_mat, node_to_index, cle_mat, fag_mat = prepare_matrices(G)
    
    # Initialize results dictionaries
    clemente_clustering = {}
    fagiolo_clustering = {}
    transitivity_num = 0
    transitivity_den = 0
        
    for node in G.nodes():
        i = node_to_index[node]
        # Calculate degree-related measures
        d_i_tot = np.sum(A_mat[i])  # Row i multiplied by unit vector
        d_i_bi = A_2_mat[i, i]  # (A^2)_ii
        
        # Calculate strength-related measures
        s_i_tot = np.sum(W_mat[i])  # Row i multiplied by unit vector
        s_i_bi = W_A_mat[i, i] / 2
        
        # Calculate Clemente clustering coefficient
        numerator_cle = cle_mat[i, i]
        denominator_cle = 2 * (s_i_tot * (d_i_tot - 1) - 2 * s_i_bi)
        
        if denominator_cle > 0:
            clemente_clustering[i] = numerator_cle / denominator_cle
        else:
            clemente_clustering[i] = 0.0
            
        # Calculate Fagiolo clustering coefficient
        numerator_fag = fag_mat[i, i]
        denominator_fag = 2 * (d_i_tot * (d_i_tot - 1) - 2 * d_i_bi)
        
        if denominator_fag > 0:
            fagiolo_clustering[i] = numerator_fag / denominator_fag
        else:
            fagiolo_clustering[i] = 0.0
        transitivity_num += numerator_fag
        transitivity_den += denominator_fag
    
    # Calculate average clustering coefficients
    avg_clemente_clustering = np.mean(list(clemente_clustering.values()))
    avg_fagiolo_clustering = np.mean(list(fagiolo_clustering.values()))
    transitivity = transitivity_num / transitivity_den
    return avg_clemente_clustering, avg_fagiolo_clustering, transitivity

# ACC, AFC, transitivity = compute_clustering(G)

# Degree distribution
def deg_dis(G):
    degree_in_dict = dict(G.in_degree())
    degree_in_list = np.array(list(degree_in_dict.values()))
    degree_out_dict = dict(G.out_degree())
    degree_out_list = np.array(list(degree_out_dict.values()))

    k_in_min = degree_in_list.min()
    fit_in_result = MLE_functions_v2.fit('Graph', degree_in_list, k_min=k_in_min, plot_type='ccdf', save=False)

    k_out_min = degree_out_list.min()
    fit_out_result = MLE_functions_v2.fit('Graph', degree_out_list, k_min=k_out_min, plot_type='ccdf', save=False)
    return fit_in_result, fit_out_result


