import cvxpy as cp
import scipy as sp
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import time
from scipy.stats import linregress
from matplotlib.ticker import NullFormatter

def generate_edges(num_of_nodes, num_of_edges):
    if num_of_edges > num_of_nodes * (num_of_nodes - 1) // 2:
        raise ValueError("Number of edges exceeds the maximum possible connections.")

    edges = set()
    while len(edges) < num_of_edges:
        node1 = random.randint(0, num_of_nodes - 1)
        node2 = random.randint(0, num_of_nodes - 1)

        if node1 != node2:
            edge = (min(node1, node2), max(node1, node2))
            edges.add(edge)

    return list(edges)


def Goemans_Williamson_max_cut(edges):
    # find number of nodes in max cut graph
    num_of_nodes = max(max(edge) for edge in edges) + 1

    # Create a symmetric matrix variable
    X = cp.Variable((num_of_nodes, num_of_nodes), symmetric=True)

    # Constraints
    constraints = [X >> 0]  # Declare matrix X to be positive semidefinite
    constraints += [X[i, i] == 1 for i in range(num_of_nodes)]  # Since we want unit vectors, set diagonals to 1

    # Objective function (Q)
    objective = cp.Maximize(sum(0.5 * (1 - X[i, j]) for i, j in edges))

    # Solve problem based on objective and constraints
    prob = cp.Problem(objective, constraints)
    prob.solve()
    X_solution = X.value

    # Finding the sqrt of the matrix X produces the vectors of the nodes given by the relaxed problem (P)
    x_projected = sp.linalg.sqrtm(X_solution)

    # Generate a random hyperplane
    u = np.random.randn(num_of_nodes)

    # Project onto the hyperplane and classify
    cut = np.sign(x_projected @ u)
    
    # cut should not have any imaginary component, so no data is lost here
    return cut.real


def brute_force_max_cut(edges, num_of_nodes):
    max_cut_size = 0
    best_cut = None

    # Iterate over all possible combinations of nodes in two sets
    for i in range(2 ** num_of_nodes):
        # Convert i to binary and pad with zeros to get node assignments
        node_assignment = np.array(list(bin(i)[2:].zfill(num_of_nodes)), dtype=int)

        # Calculate the cut size
        cut_size = sum(node_assignment[i] != node_assignment[j] for i, j in edges)

        # Update the maximum cut size and the best cut
        if cut_size > max_cut_size:
            max_cut_size = cut_size
            best_cut = node_assignment

    return best_cut


def calc_num_of_cuts(edges, cut):
    # Create a map of node to cut value (0 or 1)
    node_cut_map = {node: cut_val for node, cut_val in enumerate(cut)}

    # Count the number of edges where the nodes have different cut values
    cut_edge_count = sum(node_cut_map[i] != node_cut_map[j] for i, j in edges)
    return cut_edge_count


def plot_network(edges):
    num_of_nodes = max(max(edge) for edge in edges) + 1

    # Initialize the graph
    G = nx.Graph()

    # Add all nodes explicitly
    G.add_nodes_from(range(num_of_nodes))
    G.add_edges_from(edges)

    # Create a layout for our nodes 
    layout = nx.spring_layout(G)

    # Plot the original network
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_graph(G, layout, ax, title='')

    # Apply the cut algorithm
    cut = brute_force_max_cut(edges, num_of_nodes)
    num_of_cuts = calc_num_of_cuts(edges, cut)

    # Plot the network after the cut
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_cut_graph(G, layout, ax, cut, title='', runtime=10.00, num_of_cuts=num_of_cuts, edges=edges)

def plot_graph(G, layout, ax, title):
    ax.set_xlim([min(x for x, _ in layout.values()) - 0.1, max(x for x, _ in layout.values()) + 0.1])
    ax.set_ylim([min(y for _, y in layout.values()) - 0.1, max(y for _, y in layout.values()) + 0.1])
    nx.draw_networkx_edges(G, pos=layout, edge_color='black', ax=ax)
    nx.draw_networkx_nodes(G, pos=layout, node_color='black', node_size=300, ax=ax)
    nx.draw_networkx_labels(G, pos=layout, font_color='white', ax=ax)
    ax.set_title(title)
    plt.show()

def plot_cut_graph(G, layout, ax, cut, title, runtime, num_of_cuts, edges):
    # Map nodes to colors based on the cut
    node_colors = ['red' if c == 1 else 'blue' for c in cut]
    node_color_map = {node: color for node, color in zip(G.nodes(), node_colors)}

    ax.set_xlim([min(x for x, _ in layout.values()) - 0.1, max(x for x, _ in layout.values()) + 0.1])
    ax.set_ylim([min(y for _, y in layout.values()) - 0.1, max(y for _, y in layout.values()) + 0.1])

    # Draw edges with color based on the cut
    for edge in G.edges():
        edge_color = 'blue' if node_color_map[edge[0]] != node_color_map[edge[1]] else 'black'
        nx.draw_networkx_edges(G, pos=layout, edgelist=[edge], edge_color=edge_color, ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos=layout, node_color=node_colors, node_size=300, ax=ax)

    # Draw node labels
    nx.draw_networkx_labels(G, pos=layout, font_color='white', ax=ax)

    # Add runtime and number of cuts to the plot with padding
    plt.text(0.05, 0.95, f'Runtime: {runtime:.2f}s\nNum of Cuts: {num_of_cuts}/{len(edges)}', 
             transform=ax.transAxes, fontsize=10, verticalalignment='top', 
             bbox=dict(facecolor='white', alpha=0.5, pad=5))

    ax.set_title(title)
    plt.show()


def plot_max_cut(edges, cut, runtime, num_of_cuts):
    num_of_nodes = max(max(edge) for edge in edges) + 1

    # Initialize the graph
    G = nx.Graph()

    # Add all nodes explicitly
    G.add_nodes_from(range(num_of_nodes))
    G.add_edges_from(edges)

    # Create a layout for our nodes 
    layout = nx.spring_layout(G)

    # Initialize the figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Increase axes limits for margin
    ax.set_xlim([min(x for x, _ in layout.values()) - 0.1, max(x for x, _ in layout.values()) + 0.1])
    ax.set_ylim([min(y for _, y in layout.values()) - 0.1, max(y for _, y in layout.values()) + 0.1])

    # Map nodes to colors based on the cut
    node_colors = ['red' if c == 1 else 'blue' for c in cut]
    node_color_map = {node: color for node, color in zip(G.nodes(), node_colors)}

    # Draw nodes, ensuring all are included
    nx.draw_networkx_nodes(G, pos=layout, node_color=node_colors, ax=ax)

    # Draw edges with color based on the cut
    for edge in G.edges():
        edge_color = 'blue' if node_color_map[edge[0]] != node_color_map[edge[1]] else 'black'
        nx.draw_networkx_edges(G, pos=layout, edgelist=[edge], edge_color=edge_color, ax=ax)

    # Draw node labels
    nx.draw_networkx_labels(G, pos=layout, ax=ax)

    # Add runtime and number of cuts to the plot with padding
    plt.text(0.05, 0.95, f'Runtime: {runtime:.2f}s\nNum of Cuts: {num_of_cuts}/{len(edges)}', 
             transform=ax.transAxes, fontsize=10, verticalalignment='top', 
             bbox=dict(facecolor='white', alpha=0.5, pad=5))
    plt.show()


def measure_runtime_comparison(num_of_nodes, num_of_edges):
    edges = generate_edges(num_of_nodes, num_of_edges)

    total_brute_force_runtime = 0
    total_gw_runtime = 0
    num_runs = 5

    for _ in range(num_runs):
        start_time = time.time()
        brute_force_max_cut(edges, num_of_nodes)
        total_brute_force_runtime += time.time() - start_time

        start_time = time.time()
        Goemans_Williamson_max_cut(edges)
        total_gw_runtime += time.time() - start_time

    average_brute_force_runtime = total_brute_force_runtime / num_runs
    average_gw_runtime = total_gw_runtime / num_runs

    return average_brute_force_runtime, average_gw_runtime


def calc_num_of_edges_required(num_of_nodes):
    return int(np.ceil(((num_of_nodes)*(num_of_nodes-1))/2))


def calc_num_of_nodes_required(num_of_edges):
    # Coefficients for the quadratic equation
    a = 1
    b = -1
    c = -2 * num_of_edges

    # Calculating the discriminant
    discriminant = b**2 - 4*a*c

    # Calculate the positive root
    num_of_nodes = (-b + np.sqrt(discriminant)) / (2 * a)

    # Since number of nodes must be an integer, round up to the nearest integer
    return int(np.ceil(num_of_nodes))


def plot_complexity_runtime_graph():
    edge_range = range(2, 150, 5)
    brute_force_runtimes = []
    gw_runtimes = []

    for num_of_edges in edge_range:
        print(f"Processing {num_of_edges} edges...", end='\r')
        bf_runtime, gw_runtime = measure_runtime_comparison(calc_num_of_nodes_required(num_of_edges), num_of_edges)
        brute_force_runtimes.append(bf_runtime)
        gw_runtimes.append(gw_runtime)

    plt.figure(figsize=(10, 6))
    plt.plot(edge_range, brute_force_runtimes, label='Brute Force', color='red')
    plt.plot(edge_range, gw_runtimes, label='Goemans-Williamson', color='blue')
    plt.xlabel('Complexity (Number of Edges)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime vs Complexity for Max Cut Problems')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_complexity_runtime_bar_chart(log3=False):
    max_K = 7
    edge_values = [2**k for k in range(1, max_K + 1)]
    brute_force_runtimes = []
    gw_runtimes = []

    for num_of_edges in edge_values:
        print(f"Processing {num_of_edges} edges...")
        bf_runtime_total, gw_runtime_total = 0, 0

        bf_runtime, gw_runtime = measure_runtime_comparison(calc_num_of_nodes_required(num_of_edges), num_of_edges)
        bf_runtime_total += bf_runtime
        gw_runtime_total += gw_runtime

        # Calculate the average runtime over the iterations
        brute_force_runtimes.append(bf_runtime_total)
        gw_runtimes.append(gw_runtime_total)

    bar_width = 0.35
    index = np.array(range(len(edge_values)))

    plt.figure(figsize=(12, 6))
    plt.bar(index - bar_width/2, brute_force_runtimes, bar_width, label='Brute Force')
    plt.bar(index + bar_width/2, gw_runtimes, bar_width, label='Goemans-Williamson')

    # Adding trend lines for Brute Force and GW methods
    plt.plot(index - bar_width/2, brute_force_runtimes, color='blue', marker='o')
    plt.plot(index + bar_width/2, gw_runtimes, color='orange', marker='o')

    if log3:
        plt.yscale('log')
        ax = plt.gca()
        y_vals = ax.get_yticks()
        ax.set_yticklabels([f"{np.log(val)/np.log(3):.2f}" if val > 0 else '0' for val in y_vals])

    plt.xlabel('Number of Edges', fontsize=14)
    plt.ylabel('Average Relative Runtime' + (' (log3)' if log3 else ''), fontsize=14)
    plt.title('Runtime vs Complexity for Different Numbers of Edges with Trend Lines', fontsize=14)
    plt.xticks(index, edge_values, fontsize=14)
    plt.gca().yaxis.set_major_formatter(NullFormatter())
    plt.legend()
    plt.grid(True)
    plt.show()


def max_cuts_comparison():
    max_K = 7
    edge_values = [2**k for k in range(1, max_K + 1)]
    accuracies = []

    for num_of_edges in edge_values:
        print(f"Processing {num_of_edges} edges...", end='\r')
        accuracy_sum = 0
        run_times = 5
        for _ in range(run_times): 
            num_of_nodes = calc_num_of_nodes_required(num_of_edges)
            edges = generate_edges(num_of_nodes, num_of_edges)
            
            cuts_brute = brute_force_max_cut(edges, num_of_nodes)
            max_cuts_brute = calc_num_of_cuts(edges, cuts_brute)

            cuts_gw = Goemans_Williamson_max_cut(edges)
            max_cuts_gw = calc_num_of_cuts(edges, cuts_gw)

            # Calculate the accuracy as a percentage
            accuracy = (max_cuts_gw / max_cuts_brute) * 100
            accuracy_sum += accuracy

        # Append the average accuracy
        average_accuracy = accuracy_sum / run_times
        accuracies.append(average_accuracy)

    # Plotting as a bar chart
    plt.figure(figsize=(12, 6))

    # Positions for the bars
    blue_bar_positions = [x - 0.2 for x in range(len(edge_values))]
    orange_bar_positions = [x + 0.2 for x in range(len(edge_values))]

    # Plotting the bars
    plt.bar(orange_bar_positions, [100] * len(edge_values),  width=0.35, label='Brute Force')
    plt.bar(blue_bar_positions, accuracies, width=0.35, label='Goemans-Williamson')

    plt.axhline(y=87, color='red', linestyle='--', label='Minimum Expected Accuracy')
    plt.xlabel('Number of Edges')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy of Goemans-Williamson Algorithm Compared to Brute Force')
    plt.xticks(range(len(edge_values)), edge_values)
    plt.ylim(0, 110)  # Set y-axis limits
    plt.grid(True)
    plt.legend()
    plt.show()


def main(algorithm='gw', num_of_nodes=10):
    # calculate number of edges given nodes
    num_of_edges = calc_num_of_edges_required(num_of_nodes)

    # Generate edges for a max cut problem
    edges = generate_edges(num_of_nodes, num_of_edges)

    # start a timer
    start_time = time.time()

    if algorithm == 'gw':
        cut = Goemans_Williamson_max_cut(edges)
    else:
        cut = brute_force_max_cut(edges, num_of_nodes)

    # stop timer and calculate the runtime
    end_time = time.time()
    runtime = end_time - start_time

    num_of_cuts = calc_num_of_cuts(edges, cut)

    plot_max_cut(edges, cut, runtime, num_of_cuts)


def plot_network_edges(num_of_nodes=5):
    num_of_edges = calc_num_of_edges_required(num_of_nodes)
    edges = generate_edges(num_of_nodes, num_of_edges)
    plot_network(edges)


if __name__ == "__main__":
    # plot_network_edges(num_of_nodes=5)
    # main(algorithm='gw', num_of_nodes=98)
    # plot_complexity_runtime_bar_chart(log3=True)
    max_cuts_comparison()