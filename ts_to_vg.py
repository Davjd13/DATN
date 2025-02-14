import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler

def ts_to_vg(data: np.array, times: np.array = None, horizontal: bool = False):
    # Convert timeseries to visibility graph with DC algorithm

    if times is None:
        times = np.arange(len(data))

    network_matrix = np.zeros((len(data), len(data)))

    # DC visablity graph func
    def dc_vg(x, t, left, right, network):
        if left >= right:
            return
        k = np.argmax(x[left:right+1]) + left # Max node in left-right
        #print(left, right, k)
        for i in range(left, right+1):
            if i == k:
                continue

            visible = True
            for j in range(min(i+1, k+1), max(i, k)):
                # Visiblity check, EQ 1 from paper 
                if horizontal:
                    if x[j] >= x[i]:
                        visible = False
                        break
                else:
                    if x[j] >= x[i] + (x[k] - x[i]) * ((t[j] - t[i]) / (t[k] - t[i])):
                        visible = False
                        break

            if visible:
                network[k, i] = 1.0
                network[i, k] = 1.0
        
        dc_vg(x, t, left, k - 1, network) 
        dc_vg(x, t, k + 1, right, network) 

    dc_vg(data, times, 0, len(data) - 1, network_matrix)
    return network_matrix

def plot_ts_visibility(network: np.array, data: np.array, times: np.array = None, horizontal: bool = False):
    if times is None:
        times = np.arange(len(data))

    plt.style.use('dark_background') 
    fig, axs = plt.subplots(2, 1, sharex=True)
    # Plot connections and series
    for i in range(len(data)):
        for j in range(i, len(data)):
            if network[i, j] == 1.0:
                if horizontal:
                    axs[0].plot([times[i], times[j]], [data[i], data[i]], color='red', alpha=0.8)
                    axs[0].plot([times[i], times[j]], [data[j], data[j]], color='red', alpha=0.8)
                else:
                    axs[0].plot([times[i], times[j]], [data[i], data[j]], color='red', alpha=0.8)
    axs[0].plot(times, data)
    axs[0].bar(times, data, width=0.1)
    axs[0].get_xaxis().set_ticks(list(times))

    # Plot graph
    for i in range(len(data)):
        axs[1].plot(times[i], 0, marker='o', color='orange')

    for i in range(len(data)):
        for j in range(i, len(data)):
            if network[i, j] == 1.0:
                Path = mpath.Path
                mid_time = (times[j] + times[i]) / 2.
                diff = abs(times[j] - times[i])
                pp1 = mpatches.PathPatch(Path([(times[i], 0), (mid_time, diff), (times[j], 0)],[Path.MOVETO, Path.CURVE3, Path.CURVE3]), fc="none", transform=axs[1].transData, alpha=0.5)
                axs[1].add_patch(pp1)
    axs[1].get_yaxis().set_ticks([])
    axs[1].get_xaxis().set_ticks(list(times))
    plt.show()

def plot_ts_visibility_2(network: np.array, data: np.array, times: np.array = None, horizontal: bool = False):
    if times is None:
        times = np.arange(len(data))

    plt.style.use('default')  # Use default style for a white background
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

    axs[0].plot(times, data, color='blue', label='Time Series')  # Add line plot for the time series
    axs[0].scatter(times, data, color='cyan', edgecolor='black', s=50, label='_nolegend_')  # Small circles

    # axs[0].bar(times, data, width=0.1, color='cyan', alpha=0.8, label='Data Points')  # Bar plot for the data points
    axs[0].grid(alpha=0.3)  # Light grid for better readability
    axs[0].get_xaxis().set_ticks(list(times))

    # Plot connections and series (upper plot)
    for i in range(len(data)):
        for j in range(i, len(data)):
            if network[i, j] == 1.0:
                if horizontal:
                    axs[1].plot([times[i], times[j]], [data[i], data[i]], color='red', alpha=0.8)
                    axs[1].plot([times[i], times[j]], [data[j], data[j]], color='red', alpha=0.8)
                else:
                    axs[1].plot([times[i], times[j]], [data[i], data[j]], color='red', alpha=0.8)

    axs[1].plot(times, data, color='blue', label='Time Series')  # Add line plot for the time series
    axs[1].scatter(times, data, color='cyan', edgecolor='black', s=50, label='_nolegend_')  # Small circles
    # axs[1].bar(times, data, width=0.1, color='cyan', alpha=0.8, label='Data Points')  # Bar plot for the data points
    axs[1].grid(alpha=0.3)  # Light grid for better readability
    axs[1].get_xaxis().set_ticks(list(times))

    def calculate_node_positions(times):
        positions = {}
        for idx, time in enumerate(times):
            # Alternate between bottom and middle row
            y_pos = 1 if idx % 2 == 1 else 0
            positions[idx] = (time, y_pos)  # Use actual time values for x-position
        return positions

    # Define node positions
    node_positions = calculate_node_positions(times)

    # Plot nodes with numbers
    for node, pos in node_positions.items():
        # Draw larger circle
        circle = plt.Circle(pos, radius=0.2, color='orange', alpha=0.3)
        axs[2].add_artist(circle)
        # Add node number
        axs[2].text(pos[0], pos[1], str(node+1), 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    fontweight='bold')

    # Plot edges
    for i in range(len(data)):
        for j in range(i, len(data)):
            if network[i, j] == 1.0:
                x1, y1 = node_positions[i]
                x2, y2 = node_positions[j]
                axs[2].plot([x1, x2], [y1, y2], color='gray', alpha=0.5)

    # Customize plot
    axs[2].set_xlim(min(times)-0.5, max(times)+0.5)
    axs[2].set_ylim(-0.5, 1.5)
    # axs[1].grid(alpha=0.3)  # Light grid for better readability
    # axs[1].get_yaxis().set_ticks([])  # Remove y-axis ticks for clarity

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()

if __name__ == '__main__':

    # Example data from video
    dat = np.array([20, 40, 48, 70, 40, 60, 40, 100, 40, 80])
    #dat = np.array([.71, .53, .56, .29, .30, .77, .01, .76, .81, .71, .05, .41, .86, .79, .37, .96, .87, .06, .95, .36])

    # Cosine wave with 4 cycles (period = 12)
    #dat = np.cos( 2 * np.pi * (1/12) * np.arange(48) )

    # Daily bitcoin data, 
    #bitcoin_data = pd.read_csv('BTCUSDT86400.csv')
    # Decemeber 2022
    #dat = bitcoin_data['close'].iloc[-31:].to_numpy()
    
    # Network is the adjacency matrix
    network = ts_to_vg(dat)
  
    # Print adjacency matrix
    network = network.astype(int)
    index = list(range(len(dat)))
    index = [str(x) for x in index] 
    print("    " + " ".join(index))
    print("    " + "-" * (len(dat) * 2 - 1))
    for i in range(len(dat)):
        row = f"{i} | {str(network[:, i])[1:-1]}"
        print(row)

    plot_ts_visibility(network, dat, horizontal=False)

def ts_to_cross_vg(data1: np.array, times1: np.array = None, data2: np.array = None, times2: np.array = None, horizontal: bool = False):
    # Default times if not provided
    if times1 is None:
        times1 = np.arange(len(data1))
    if times2 is None:
        times2 = np.arange(len(data2))
    scaler = MinMaxScaler()
    data1_scaled = scaler.fit_transform(data1.reshape(-1, 1)).flatten()
    data2_scaled = scaler.fit_transform(data2.reshape(-1, 1)).flatten()

    # Compute the maximum values for combined datasets
    max_data = np.maximum(data1_scaled, data2_scaled)
    times_max = np.arange(len(max_data))  # Should be consistent with the times for both data1 and data2
    
    # Initialize the network matrix
    network_matrix = np.zeros((len(data1_scaled) + len(data2_scaled), len(data1_scaled) + len(data2_scaled)))

    # Loop through all combinations of points in data1 and data2
    for i, (x1, t1) in enumerate(zip(data1_scaled, times1)):
        for j, (x2, t2) in enumerate(zip(data2_scaled, times2)):
            if i == j:
                continue  # Skip self-loops

            visible = True  # Assume visibility unless proven otherwise

            # Immediate neighbors are always visible
            if abs(i - j) == 1:
                visible = True

            # If the points are not immediate neighbors, check visibility
            elif abs(i - j) > 1:
                # Determine the range of intermediate indices
                min_index = min(i, j)
                max_index = max(i, j)

                # Get intermediate data points and times
                k_between = max_data[min_index + 1:max_index]
                times_between = times_max[min_index + 1:max_index]

                # Visibility check
                for k, tk in zip(k_between, times_between):
                    if horizontal:
                        # Horizontal visibility: check if any intermediate point is greater or equal to the minimum of the endpoints
                        if k >= min(x1, x2):
                            visible = False
                            break
                    else:
                        # Standard visibility: use the linear visibility equation
                        if k >= x1 + (x2 - x1) * (tk - t1) / (t2 - t1):
                            visible = False
                            break

            # If the points are visible, update the network matrix
            if visible:
                network_matrix[i, j + len(data1_scaled)] = 1.0  # Visibility from data1 to data2
                network_matrix[j + len(data1_scaled), i] = 1.0  # Visibility from data2 to data1

    return network_matrix

def plot_cross_visibility(
    network: np.array, network1: np.array, network2: np.array,
    data1: np.array, data2: np.array, 
    times1: np.array = None, times2: np.array = None, 
    horizontal: bool = False
):
    if times1 is None:
        times1 = np.arange(len(data1))
    if times2 is None:
        times2 = np.arange(len(data2))
    
    def original(data1, data2, times1, times2):
        plt.style.use('default')
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # Original time series on axs[0]
        axs[0].plot(times1, data1, color='blue', label='Time Series 1')
        axs[0].scatter(times1, data1, color='blue', edgecolor='black', s=40, label='_nolegend_')  # Small circles
        axs[0].plot(times2, data2, color='orange', label='Time Series 2')
        axs[0].scatter(times2, data2, color='orange', edgecolor='black', s=40, label='_nolegend_')  # Small circles
        axs[0].grid(alpha=0.3)
        axs[0].legend()

        # Scaled time series on axs[1]
        scaler = MinMaxScaler()
        data1_scaled = scaler.fit_transform(data1.reshape(-1, 1)).flatten()
        data2_scaled = scaler.fit_transform(data2.reshape(-1, 1)).flatten()

        axs[1].scatter(times1, data1_scaled, color='blue', edgecolor='black', s=50, label='_nolegend_')  # Small circles
        axs[1].bar(times1, data1_scaled, width=0.03, color='cyan', alpha=0.8)
        axs[1].scatter(times2, data2_scaled, color='orange', edgecolor='black', s=50, label='_nolegend_')  # Small circles
        axs[1].bar(times2, data2_scaled, width=0.03, color='yellow', alpha=0.8)
        axs[1].grid(alpha=0.3)
        axs[0].get_xaxis().set_ticks(list(times1))
        axs[1].get_xaxis().set_ticks(list(times1))

        plt.tight_layout()
        plt.show()

    def connected(network, data1, data2, times1, times2, horizontal: bool = False):
        plt.style.use('default')
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        
        # Original time series on axs[0]
        axs[0].plot(times1, data1, color='blue', label='Time Series 1')
        axs[0].scatter(times1, data1, color='blue', edgecolor='black', s=40, label='_nolegend_')  # Small circles
        axs[0].plot(times2, data2, color='orange', label='Time Series 2')
        axs[0].scatter(times2, data2, color='orange', edgecolor='black', s=40, label='_nolegend_')  # Small circles
        axs[0].grid(alpha=0.3)
        axs[0].legend()

        # Scaled time series on axs[1]
        scaler = MinMaxScaler()
        data1_scaled = scaler.fit_transform(data1.reshape(-1, 1)).flatten()
        data2_scaled = scaler.fit_transform(data2.reshape(-1, 1)).flatten()

        axs[1].scatter(times1, data1_scaled, color='blue', edgecolor='black', s=50, label='_nolegend_')  # Small circles
        axs[1].bar(times1, data1_scaled, width=0.03, color='cyan', alpha=0.8)
        axs[1].scatter(times2, data2_scaled, color='orange', edgecolor='black', s=50, label='_nolegend_')  # Small circles
        axs[1].bar(times2, data2_scaled, width=0.03, color='yellow', alpha=0.8)

        # Draw edges based on the network
        for i, x1 in enumerate(data1_scaled):
            for j, x2 in enumerate(data2_scaled):
                if network[i, j + len(data1)] == 1.0:  # Visibility from data1 to data2
                    index_diff = abs(i - j)
                    if index_diff == 1:
                    # Adjacent nodes - gray dashed line
                        axs[0].plot(
                        [times1[i], times2[j]],
                        [data1[i], data2[j]],
                        color='gray',
                        alpha=0.5,
                        linestyle='--')
                    else:
                        # Non-adjacent nodes - red dashed line
                        axs[0].plot(
                        [times1[i], times2[j]],
                        [data1[i], data2[j]],
                        color='red',
                        alpha=0.5,
                        linestyle='--')                  
                    if horizontal:
                        # Horizontal line connections
                        axs[1].plot(
                            [times1[i], times2[j]], [x1, x1], color='red', alpha=0.8, linestyle='--'
                        )
                        axs[1].plot(
                            [times1[i], times2[j]], [x2, x2], color='green', alpha=0.8, linestyle='--'
                        )
                    else:
                        # Diagonal line connections
                        axs[1].plot(
                            [times1[i], times2[j]], [x1, x2], color='purple', alpha=0.8
                        )

        axs[1].grid(alpha=0.3)
        axs[0].get_xaxis().set_ticks(list(times1))
        axs[1].get_xaxis().set_ticks(list(times1))
        
        plt.tight_layout()
        plt.show()

    def plot_cross_visibility_2(network, network1, network2, data1, data2, times1, times2):
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        def calculate_node_positions1(times1):
            positions1 = {}
            for idx, time in enumerate(times1):
                # Alternate between bottom and middle row
                y_pos = 4 if idx % 2 == 1 else 3
                positions1[idx] = (time, y_pos)  # Use actual time values for x-position
            return positions1
        def calculate_node_positions2(times2):
            positions2 = {}
            for idx, time in enumerate(times2):
                y_pos = 1 if idx % 2 == 1 else 0
                positions2[idx] = (time, y_pos)
            return positions2

        # Define node positions
        node_positions1 = calculate_node_positions1(times1)
        node_positions2 = calculate_node_positions2(times2)

        # Plot nodes with numbers
        for node, pos in node_positions1.items():
            # Draw larger circle
            circle = plt.Circle(pos, radius=0.2, color='blue', alpha=0.3)
            ax.add_artist(circle)
            # Add node number
            ax.text(pos[0], pos[1], str(node+1), 
                        horizontalalignment='center', 
                        verticalalignment='center',
                        fontweight='bold')

        for node, pos in node_positions2.items():
            # Draw larger circle
            circle = plt.Circle(pos, radius=0.2, color='orange', alpha=0.3)
            ax.add_artist(circle)
            # Add node number
            ax.text(pos[0], pos[1], str(node+1), 
                        horizontalalignment='center', 
                        verticalalignment='center',
                        fontweight='bold')
        # Plot edges
        for i in range(len(data1)):
            for j in range(len(data2)):
                if network[i, j + len(data1)] == 1.0:
                    x1, y1 = node_positions1[i]
                    x2, y2 = node_positions2[j]
                    index_diff = abs(i - j)
                    if index_diff == 1:
                    # Adjacent nodes - gray dashed line
                        ax.plot([x1, x2], [y1, y2], 
                            color='gray', 
                            alpha=0.5,
                            linestyle='--',
                            linewidth=1.5)
                    else:
                        # Non-adjacent nodes - red dashed line
                        ax.plot([x1, x2], [y1, y2], 
                            color='red', 
                            alpha=0.5,
                            linestyle='--',
                            linewidth=1.5)

        for i in range(len(data1)):
            for j in range(i, len(data1)):
                if network1[i, j] == 1.0:
                    x1, y1 = node_positions1[i]
                    x2, y2 = node_positions1[j]
                    ax.plot([x1, x2], [y1, y2], color='blue', alpha=0.5)

        for i in range(len(data2)):
            for j in range(i, len(data2)):
                if network2[i, j] == 1.0:
                    x1, y1 = node_positions2[i]
                    x2, y2 = node_positions2[j]
                    ax.plot([x1, x2], [y1, y2], color='orange', alpha=0.5)
        ax.set_ylim(-0.5, 4.5)
        plt.tight_layout()
        plt.show()

    def Multiplex(network1, network2, data1, data2, times1, times2):
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        def calculate_node_positions1(times1):
            positions1 = {}
            for idx, time in enumerate(times1):
                # Alternate between bottom and middle row
                y_pos = 4 if idx % 2 == 1 else 3
                positions1[idx] = (time, y_pos)  # Use actual time values for x-position
            return positions1
        def calculate_node_positions2(times2):
            positions2 = {}
            for idx, time in enumerate(times2):
                y_pos = 1 if idx % 2 == 1 else 0
                positions2[idx] = (time, y_pos)
            return positions2

        # Define node positions
        node_positions1 = calculate_node_positions1(times1)
        node_positions2 = calculate_node_positions2(times2)

        # Plot nodes with numbers
        for node, pos in node_positions1.items():
            # Draw larger circle
            circle = plt.Circle(pos, radius=0.2, color='blue', alpha=0.3)
            ax.add_artist(circle)
            # Add node number
            ax.text(pos[0], pos[1], str(node+1), 
                        horizontalalignment='center', 
                        verticalalignment='center',
                        fontweight='bold')

        for node, pos in node_positions2.items():
            # Draw larger circle
            circle = plt.Circle(pos, radius=0.2, color='orange', alpha=0.3)
            ax.add_artist(circle)
            # Add node number
            ax.text(pos[0], pos[1], str(node+1), 
                        horizontalalignment='center', 
                        verticalalignment='center',
                        fontweight='bold')
        for i in range(len(data1)):
            for j in range(i, len(data1)):
                if network1[i, j] == 1.0:
                    x1, y1 = node_positions1[i]
                    x2, y2 = node_positions1[j]
                    ax.plot([x1, x2], [y1, y2], color='blue', alpha=0.5)

        for i in range(len(data2)):
            for j in range(i, len(data2)):
                if network2[i, j] == 1.0:
                    x1, y1 = node_positions2[i]
                    x2, y2 = node_positions2[j]
                    ax.plot([x1, x2], [y1, y2], color='orange', alpha=0.5)
        for i in range(len(data1)):
            for j in range(len(data2)):
                if i == j:
                    x1, y1 = node_positions1[i]
                    x2, y2 = node_positions2[j]
                    ax.plot([x1, x2], [y1, y2], color='gray', alpha=0.5, linestyle='--')

        ax.set_ylim(-0.5, 4.5)
        plt.tight_layout()
        plt.show()

    original(data1, data2, times1, times2)
    connected(network, data1, data2, times1, times2, horizontal)
    plot_cross_visibility_2(network, network1, network2, data1, data2, times1, times2)
    Multiplex(network1, network2, data1, data2, times1, times2)