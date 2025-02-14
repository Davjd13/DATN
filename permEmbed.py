import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score
from scipy.signal import argrelextrema
from itertools import permutations
from sklearn.neighbors import NearestNeighbors

def perm_embedding(X, dim, lag):
    """
    Constructs the permutation embedding from a given time-series.
    
    Arguments:
        X: array-like
            A one-dimensional Numpy array of numerical values
        dim: int
            The embedding dimension - number of points to consider for each permutation
        lag: int
            The embedding lag - time delay between consecutive points
            
    Returns:
        series: array
            Array of strings representing the permutation patterns
            Each string contains the relative ordering of 'dim' points
    """
    # Step 1: Create overlapping time-delayed vectors
    n_vectors = X.shape[0] - lag * (dim-1)
    timestep = np.zeros((n_vectors, dim))
    
    # Build the delay vectors
    for t in range(n_vectors):
        for d in range(dim):
            timestep[t][d] = X[t + d*lag]
    
    # Step 2: Convert each vector into its ordinal pattern
    # First get the ranking of each element within its vector
    # Then get the ranking of these rankings to handle ties consistently
    sort = np.argsort(np.argsort(timestep, axis=1)) + 1
    
    # Step 3: Find unique patterns
    unique = np.unique(sort, axis=0)
    
    # Step 4: Create a mapping from patterns to string representations
    # Convert each pattern to bytes for dictionary key (arrays aren't hashable)
    pattern_to_string = {
        unique[i].tobytes(): "".join(str(x) for x in unique[i])
        for i in range(unique.shape[0])
    }
    
    # Step 5: Convert each pattern in the series to its string representation
    # Note: we exclude the last pattern as in the original code
    series = np.zeros(sort.shape[0], dtype=object)
    for s in range(sort.shape[0]):
        series[s] = pattern_to_string[sort[s].tobytes()]
        
    return series
def OPN(ts, dim, lag):
    """
    Constructs the time-lagged, ordinal partition network from a given time-series.
    
    Arguments:
        ts: array-like
            A one-dimensional Numpy array of numerical values 
        dim: int
            The embedding dimension
        lag: int
            The embedding lag
    
    Returns:
        G: networkx.DiGraph
            A directed network representing transitions between ordinal patterns
        series: array
            The permutation embedding of the original series ts
    """
    # Step 1: Create overlapping time-delayed vectors
    n_vectors = ts.shape[0] - lag * (dim-1)
    timestep = np.zeros((n_vectors, dim))
    
    # Build the delay vectors
    for t in range(n_vectors):
        for d in range(dim):
            timestep[t][d] = ts[t + d*lag]
    
    # Step 2: Convert each vector into its ordinal pattern (1-based)
    sort = np.argsort(np.argsort(timestep, axis=1)) + 1
    unique = np.unique(sort, axis=0)
    
    # Step 3: Create mapping from patterns to string representations
    pattern_to_string = {
        unique[i].tobytes(): "".join(str(x) for x in unique[i])
        for i in range(unique.shape[0])
    }
    
    # Step 4: Initialize directed graph
    G = nx.Graph()
    
    # Add all unique patterns as nodes
    G.add_nodes_from(list(pattern_to_string.values()))
    
    # Step 5: Create edges from consecutive patterns
    edges = []
    weights = []
    series = np.zeros(sort.shape[0]-1, dtype=object)
    
    for s in range(sort.shape[0]-1):
        source = pattern_to_string[sort[s].tobytes()]
        target = pattern_to_string[sort[s+1].tobytes()]
        edges.append((source, target))
        series[s] = source
    
    # Step 6: Add edges to graph
    # NetworkX handles edge weights differently from igraph
    edge_counts = {}
    for edge in edges:
        if edge in edge_counts:
            edge_counts[edge] += 1
        else:
            edge_counts[edge] = 1
    
    # Add weighted edges to graph
    for (source, target), weight in edge_counts.items():
        G.add_edge(source, target, weight=weight)
    
    return G, series

def OPN1(ts, dim, lag):
    """
    Constructs the time-lagged, ordinal partition network from a given time-series.
    
    Arguments:
        ts: array-like
            A one-dimensional Numpy array of numerical values 
        dim: int
            The embedding dimension
        lag: int
            The embedding lag
    
    Returns:
        G: networkx.DiGraph
            A directed network representing transitions between ordinal patterns
        series: array
            The permutation embedding of the original series ts
        miss_perm: list
            A list of missing ordinal patterns that do not appear in the time series
    """
    # Step 1: Create overlapping time-delayed vectors
    n_vectors = ts.shape[0] - lag * (dim-1)
    timestep = np.zeros((n_vectors, dim))
    
    # Build the delay vectors
    for t in range(n_vectors):
        for d in range(dim):
            timestep[t][d] = ts[t + d*lag]
    
    # Step 2: Convert each vector into its ordinal pattern (1-based)
    sort = np.argsort(np.argsort(timestep, axis=1)) + 1
    unique = np.unique(sort, axis=0)
    
    # Step 3: Create mapping from patterns to string representations
    pattern_to_string = {
        unique[i].tobytes(): "".join(str(x) for x in unique[i])
        for i in range(unique.shape[0])
    }
    
    # Step 4: Initialize directed graph
    G = nx.DiGraph()
    
    # Add all unique patterns as nodes
    G.add_nodes_from(list(pattern_to_string.values()))
    
    # Step 5: Create edges from consecutive patterns
    edges = []
    weights = []
    series = np.zeros(sort.shape[0]-1, dtype=object)
    
    for s in range(sort.shape[0]-1):
        source = pattern_to_string[sort[s].tobytes()]
        target = pattern_to_string[sort[s+1].tobytes()]
        edges.append((source, target))
        series[s] = source
    
    # Step 6: Add edges to graph
    # NetworkX handles edge weights differently from igraph
    edge_counts = {}
    for edge in edges:
        if edge in edge_counts:
            edge_counts[edge] += 1
        else:
            edge_counts[edge] = 1
    
    # Normalize edge weights
    total_weight = sum(edge_counts.values())
    for (source, target), weight in edge_counts.items():
        normalized_weight = weight / total_weight
        G.add_edge(source, target, weight=normalized_weight)
    
    # Step 7: Identify missing patterns
    all_patterns = set([''.join(map(str, p)) for p in permutations(range(1, dim+1))])
    observed_patterns = set(pattern_to_string.values())
    miss_perm = sorted(all_patterns - observed_patterns)
    
    return G, series, miss_perm

def OPN2(ts, dim, lag, verbose=True):
    """
    Constructs the time-lagged, ordinal partition network with explicit node mapping.
    
    Arguments:
        ts: array-like
            A one-dimensional Numpy array of numerical values 
        dim: int
            The embedding dimension
        lag: int
            The embedding lag
        verbose: bool
            If True, prints the mapping between patterns and matrix indices
    
    Returns:
        G: networkx.DiGraph
            A directed network representing transitions between ordinal patterns
        series: array
            The permutation embedding of the original series ts
        node_mapping: dict
            Mapping between ordinal patterns and matrix indices
    """
    # Steps 1-2: Create time-delayed vectors and get ordinal patterns
    n_vectors = ts.shape[0] - lag * (dim-1)
    timestep = np.zeros((n_vectors, dim))
    
    for t in range(n_vectors):
        for d in range(dim):
            timestep[t][d] = ts[t + d*lag]
    
    sort = np.argsort(np.argsort(timestep, axis=1)) + 1
    unique = np.unique(sort, axis=0)
    
    # Create pattern to string mapping
    pattern_to_string = {
        unique[i].tobytes(): "".join(str(x) for x in unique[i])
        for i in range(unique.shape[0])
    }
    
    # Create ordered list of unique patterns
    unique_patterns = sorted(list(set(pattern_to_string.values())))
    
    # Create explicit mapping between patterns and indices
    node_mapping = {pattern: idx for idx, pattern in enumerate(unique_patterns)}
    
    # Initialize graph
    G = nx.DiGraph()
    
    # Add nodes with explicit indices
    for pattern, idx in node_mapping.items():
        G.add_node(pattern, index=idx)
    
    # Create edges and series
    edges = []
    series = np.zeros(sort.shape[0]-1, dtype=object)
    
    for s in range(sort.shape[0]-1):
        source = pattern_to_string[sort[s].tobytes()]
        target = pattern_to_string[sort[s+1].tobytes()]
        edges.append((source, target))
        series[s] = source
    
    # Add weighted edges
    edge_counts = {}
    for edge in edges:
        if edge in edge_counts:
            edge_counts[edge] += 1
        else:
            edge_counts[edge] = 1
    
    for (source, target), weight in edge_counts.items():
        G.add_edge(source, target, weight=weight)
    
    if verbose:
        print("\nNode mapping (pattern -> matrix index):")
        for pattern, idx in node_mapping.items():
            print(f"{pattern}: {idx}")
            
    return G, series, node_mapping

# Example usage and visualization
def visualize_adjacency(G, node_mapping):
    """
    Visualizes the adjacency matrix with pattern labels.
    """
    import matplotlib.pyplot as plt
    
    # Get adjacency matrix
    A = nx.adjacency_matrix(G).todense()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(A, cmap='Blues')
    
    # Add pattern labels
    patterns = sorted(node_mapping.keys())
    plt.xticks(range(len(patterns)), patterns, rotation=45)
    plt.yticks(range(len(patterns)), patterns)
    
    # Add colorbar and values
    plt.colorbar()
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i,j] > 0:
                plt.text(j, i, int(A[i,j]), ha='center', va='center')
    
    plt.title('Adjacency Matrix with Pattern Labels')
    plt.tight_layout()
    plt.show()

def optimal_lag(X, lrange, step):
    """
    Returns the embedding lag corresponding to the first zero of the autocorrelation function.
    
    Arguments:
        X: array-like
            The time-series data
        lrange: int
            The range of possible embedding lags to try, from 1...lrange
        step: int
            Step size for sampling the lag range. Larger steps are computationally faster
            but give less precise results
    
    Returns:
        lag: int
            The lag that gives the first zero of the autocorrelation function
    """
    # Initialize autocorrelation array
    autocorr = np.ones(lrange // step)
    pos = True
    extrema = -1
    
    # Calculate autocorrelation for different lags
    for i in range(1, len(autocorr)):
        if pos:
            # Calculate Pearson correlation between series and lagged series
            correlation = pearsonr(X[:-step*i], X[step*i:])[0]
            autocorr[i] = correlation
            
            # Check for sign change from positive to negative
            if autocorr[i] < 0 and autocorr[i-1] > 0:
                pos = False
                extrema = i-1
                
    return extrema * step

def calculate_dmax(X, lag):
    """
    Calculate maximum possible embedding dimension based on time series length and lag.
    
    Arguments:
        X: array-like
            The time-series data
        lag: int
            The embedding lag
            
    Returns:
        dmax: int
            Maximum possible embedding dimension
    """
    # From n_vectors = X.shape[0] - lag*(dmax-1) > 1
    # Solve for dmax:
    # X.shape[0] - lag*(dmax-1) = 1
    # -lag*dmax + lag = -X.shape[0] + 1
    # dmax = (X.shape[0] + lag - 1)/lag
    dmax = (X.shape[0] + lag - 1)//lag
    return dmax

def optimal_dim(X, lag, drange=None):
    """
    Returns the embedding dimension that maximizes the variance in the degree 
    distribution of the resulting Ordinal Partition Network.
    
    Arguments:
        X: array-like
            The time-series data
        lag: int
            The embedding lag
        drange: int, optional
            The range of possible embedding dimensions to try.
            If None, automatically calculated based on time series length
            
    Returns:
        dim: int
            The embedding dimension that maximized the variance in the 
            degree distribution of the resulting network
    """
    # Calculate maximum possible dimension
    dmax = calculate_dmax(X, lag)
    
    # If drange is not specified or is too large, use dmax
    if drange is None or drange > dmax:
        drange = dmax
        
    # Ensure minimum dimension of 2
    if drange < 2:
        raise ValueError(f"Time series too short for given lag. Maximum possible dimension is {dmax}")
        
    # Initialize variance array
    var = np.zeros(drange)
    
    # Calculate variance of degree distribution for each dimension
    for i in range(2, drange):
        # Verify we can create vectors of this dimension
        n_vectors = X.shape[0] - lag * (i-1)
        if n_vectors <= 0:
            print(f"Warning: Dimension {i} results in {n_vectors} vectors. Stopping here.")
            break
            
        # Create OPN for current dimension
        G, _ = OPN(X, i, lag)
        
        # Calculate variance of degree distribution
        degrees = [d for _, d in G.degree()]
        var[i] = np.var(degrees)
        
    # Return dimension with maximum variance
    optimal_d = np.argmax(var)
    
    return optimal_d

def mutual_info_lag(X, lrange, step):
    """
    Returns lag corresponding to first minimum of mutual information
    using optimal bin size based on data length
    """
    X = np.asarray(X).ravel()
    
    mi = np.zeros(lrange // step)
    bins = int(np.sqrt(len(X)/5))  # optimal bin size
    
    for i in range(1, len(mi)):
        # Create lagged series
        x1 = X[:-step*i]
        x2 = X[step*i:]
        
        # Reshape arrays for histogram2d
        x1 = np.asarray(x1).reshape(-1)
        x2 = np.asarray(x2).reshape(-1)
        
        # Calculate MI using sklearn
        c_xy = np.histogram2d(x1, x2, bins=bins)[0]
        mi[i] = mutual_info_score(None, None, contingency=c_xy)
        
        # Check for first minimum
        if i > 1 and mi[i] > mi[i-1] and mi[i] != 0:
            return (i-1) * step
            
    return len(mi) * step

# not finished
def optimal_dim_fnn(X, lag=1, max_dim=10, threshold=2.0, rtol=0.05):
    """
    Determines optimal embedding dimension using False Nearest Neighbors method.
    
    Arguments:
        X: array-like
            The time series data
        lag: int, optional (default=1)
            Time delay between successive embedding dimensions
        max_dim: int, optional (default=10)
            Maximum embedding dimension to test
        threshold: float, optional (default=2.0)
            Threshold for identifying false neighbors
        rtol: float, optional (default=0.05)
            Tolerance for considering FNN ratio as stabilized
            
    Returns:
        optimal_dim: int
            Optimal embedding dimension
        fnn_ratios: array
            Ratio of false nearest neighbors for each dimension
    """
    X = np.asarray(X).flatten()
    fnn_ratios = np.zeros(max_dim)

    for dim in range(1, max_dim + 1):
        try:
            vectors = perm_embedding(X, dim, lag)
            if vectors.ndim == 1:
                vectors = vectors.reshape(-1, 1)  # Ensure 2D shape
        except ValueError as e:
            print(f"Stopping at dimension {dim}: {e}")
            break

        if vectors.shape[0] < 2:
            print(f"Warning: Not enough vectors for dimension {dim}")
            break

        # Find nearest neighbors in current dimension
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(vectors)
        distances, indices = nbrs.kneighbors(vectors)

        # Get the second nearest neighbor (first is the point itself)
        nearest_neighbors = indices[:, 1]
        dist_current = distances[:, 1]  # Distance to nearest neighbor

        # Skip if this is the last dimension
        if dim == max_dim:
            break

        # Create embedding for next dimension
        try:
            vectors_next = perm_embedding(X, dim + 1, lag)
            if vectors_next.ndim == 1:
                vectors_next = vectors_next.reshape(-1, 1)
        except ValueError as e:
            print(f"Stopping at dimension {dim + 1}: {e}")
            break

        # Calculate number of false neighbors
        false_neighbors = 0
        valid_points = 0

        for i in range(len(vectors_next) - 1):
            if dist_current[i] == 0:  # Avoid division by zero
                continue

            dist_next = np.linalg.norm(vectors_next[i] - vectors_next[nearest_neighbors[i]])

            if dist_next / dist_current[i] > threshold:
                false_neighbors += 1
            valid_points += 1

        # Compute ratio of false neighbors
        fnn_ratios[dim - 1] = false_neighbors / valid_points if valid_points > 0 else 1.0

        # Debugging print (optional)
        # print(f"dim={dim}, false_neighbors={false_neighbors}, valid_points={valid_points}, fnn_ratio={fnn_ratios[dim - 1]}")

        # Check if FNN ratio has stabilized
        if dim > 1 and abs(fnn_ratios[dim - 1] - fnn_ratios[dim - 2]) < rtol:
            optimal_dim = dim
            break
    else:
        # If no stabilization found, use dimension with minimum FNN ratio
        optimal_dim = np.argmin(fnn_ratios[:dim]) + 1

    return optimal_dim, fnn_ratios


# not suitable
def extrema_embed(X, dim, lag):

    
    """
    Constructs D! separate time series using extrema values from the original time series X.
    
    Arguments:
        X: array-like
            A one-dimensional Numpy array of numerical values
        dim: int
            The embedding dimension - number of points to consider for each permutation
        lag: int
            The embedding lag - time delay between consecutive points
    """
    # Step 1: Get permutation patterns for the time series
    patterns = perm_embedding(X, dim, lag)
    
    # Step 2: Generate all possible permutation patterns for dimension dim
    all_perms = [''.join(map(str, p)) for p in permutations(range(1, dim + 1))]
    
    # Step 3: Initialize time series for each pattern
    pattern_series = {perm: np.zeros(len(patterns)) for perm in all_perms}
    
    # Step 4: Process each pattern in the original series
    for t, pattern in enumerate(patterns):
        # Convert current pattern to array for extrema analysis
        pattern_array = np.array([int(c) for c in pattern])
        
        # Find local minima and maxima
        min_idx = argrelextrema(pattern_array, np.less)[0]
        max_idx = argrelextrema(pattern_array, np.greater)[0]
        
        # Find absolute min and max indices in the pattern
        abs_min_index = np.argmin(pattern_array)
        abs_max_index = np.argmax(pattern_array)
        
        # Calculate the actual positions in the original time series
        # Each position needs to account for both the current position t and the lag
        actual_min_pos = t + abs_min_index * lag
        actual_max_pos = t + abs_max_index * lag
        
        if len(min_idx) == 0 and len(max_idx) == 0:
            # Use values from original time series X
            value = X[actual_max_pos] - X[actual_min_pos]
            pattern_series[pattern][t] = value
            
        # Add other cases here based on your requirements
        elif len(min_idx) == 1 and len(max_idx) == 0:
            local_min_pos = t + min_idx[0] * lag
            value = X[actual_max_pos] - X[local_min_pos]
            pattern_series[pattern][t] = value
        
        elif len(min_idx) == 0 and len(max_idx) == 1:
            local_max_pos = t + max_idx[0] * lag
            value = X[local_max_pos] - X[actual_min_pos]
            pattern_series[pattern][t] = value
            
    return pattern_series

