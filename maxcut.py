import cvxpy as cp
import scipy as sp
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from itertools import product


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
    num_of_nodes = max(max(edge) for edge in edges) + 1

    # Create a symmetric matrix variable
    X = cp.Variable((num_of_nodes, num_of_nodes), symmetric=True)

    # Constraints
    constraints = [X >> 0]  # X is positive semidefinite
    constraints += [X[i, i] == 1 for i in range(num_of_nodes)]  # Diagonals must be 1

    # Objective function (Q)
    objective = cp.Maximize(sum(0.5 * (1 - X[i, j]) for i, j in edges))

    # Solve problem based on objective and constraints
    prob = cp.Problem(objective, constraints)
    prob.solve()
    X_solution = X.value

    # Generate a random hyperplane
    u = np.random.randn(num_of_nodes)

    # Project onto the hyperplane and classify
    # Finding the sqrt of the matrix X produces the vectors of the nodes given by the relaxed problem (P)
    x_projected = sp.linalg.sqrtm(X_solution)
    cut = np.sign(x_projected @ u)
    cut = cut.real.astype(np.int32)

    return cut


def brute_force_max_cut(edges):
    num_of_nodes = max(max(edge) for edge in edges) + 1
    max_cut_size = 0
    best_cut = None

    # Iterate over all possible combinations of nodes in two sets
    for node_assignment in product([0, 1], repeat=num_of_nodes):
        cut_size = sum(node_assignment[i] != node_assignment[j] for i, j in edges)

        if cut_size > max_cut_size:
            max_cut_size = cut_size
            best_cut = node_assignment

    return np.array(best_cut)


def calc_num_of_cuts(edges, cut):
    # Create a map of node to cut value (0 or 1)
    node_cut_map = {node: cut_val for node, cut_val in enumerate(cut)}

    # Count the number of edges where the nodes have different cut values
    cut_edge_count = sum(node_cut_map[i] != node_cut_map[j] for i, j in edges)
    return cut_edge_count


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
    num_runs = 1

    for _ in range(num_runs):
        # start_time = time.time()
        # brute_force_max_cut(edges)
        # total_brute_force_runtime += time.time() - start_time
        total_brute_force_runtime = 1

        start_time = time.time()
        Goemans_Williamson_max_cut(edges)
        total_gw_runtime += time.time() - start_time

    average_brute_force_runtime = total_brute_force_runtime / num_runs
    average_gw_runtime = total_gw_runtime / num_runs

    return average_brute_force_runtime, average_gw_runtime


def plot_complexity_runtime_graph():
    num_of_nodes = 50

    edge_range = range(2, 300, 2)
    brute_force_runtimes = []
    gw_runtimes = []

    for num_of_edges in edge_range:
        print(f"Processing {num_of_edges} edges...", end='\r')
        bf_runtime, gw_runtime = measure_runtime_comparison(num_of_nodes, num_of_edges)
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

def plot_complexity_runtime_bar_chart():
    num_of_nodes = 90

    edge_values = [1, 3, 10, 30, 100, 300, 1000, 3000]
    brute_force_runtimes = []
    gw_runtimes = []

    for num_of_edges in edge_values:
        print(f"Processing {num_of_edges} edges...", end='\r')
        bf_runtime, gw_runtime = measure_runtime_comparison(num_of_nodes, num_of_edges)
        brute_force_runtimes.append(bf_runtime)
        gw_runtimes.append(gw_runtime)

    # Create a bar chart
    bar_width = 0.35
    index = range(len(edge_values))

    plt.figure(figsize=(12, 6))
    plt.bar([i - bar_width/2 for i in index], brute_force_runtimes, bar_width, label='Brute Force')
    plt.bar([i + bar_width/2 for i in index], gw_runtimes, bar_width, label='Goemans-Williamson')

    plt.xlabel('Number of Edges')
    plt.ylabel('Average Runtime (seconds)')
    plt.title('Runtime vs Complexity for Different Numbers of Edges')
    plt.xticks(index, edge_values)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_complexity_runtime_bar_chart()

    # Call the function to plot the graph
    # plot_complexity_runtime_graph()
    
    # # Nodes and edges
    # num_of_nodes = 4
    # num_of_edges = 6

    # # Generate edges for a max cut problem
    # edges = generate_edges(num_of_nodes, num_of_edges)

    # # start a timer
    # start_time = time.time()

    # # cut = Goemans_Williamson_max_cut(edges)
    # cut = brute_force_max_cut(edges)

    # # stop timer and calculate the runtime
    # end_time = time.time()
    # runtime = end_time - start_time

    # num_of_cuts = calc_num_of_cuts(edges, cut)

    # plot_max_cut(edges, cut, runtime, num_of_cuts)