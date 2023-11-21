import cvxpy as cp
import scipy as sp
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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


num_of_nodes = 8
num_of_edges = 16

# Generate edges for a max cut problem
edges = generate_edges(num_of_nodes, num_of_edges)
print(edges)

# Create a symmetric matrix variable
X = cp.Variable((num_of_nodes, num_of_nodes), symmetric=True)

# Constraints
constraints = [X >> 0]  # X is positive semidefinite
constraints += [X[i, i] == 1 for i in range(num_of_nodes)]  # Diagonals must be 1

# Objective function (from paper)
objective = cp.Maximize(sum(0.5 * (1 - X[i, j]) for i, j in edges))

# Solve problem based on objective and constraints
prob = cp.Problem(objective, constraints)
prob.solve()
X_solution = X.value

# Generate a random hyperplane
u = np.random.randn(num_of_nodes)

# Project onto the hyperplane and classify
x_projected = sp.linalg.sqrtm(X_solution)
cut = np.sign(x_projected @ u)
cut = cut.real.astype(np.int32)

# Initialize the graph
G = nx.Graph()
G.add_edges_from(edges)

# Create a layout for our nodes 
layout = nx.spring_layout(G)

# Initialize the figure
fig, ax = plt.subplots(figsize=(8, 5))

# Draw the graph
colors = ['red' if c == 1 else 'blue' for c in cut]
nx.draw(G, pos=layout, node_color=colors, with_labels=True, ax=ax)

# # Drawing a dotted line for the max cut
# for edge in edges:
#     if cut[edge[0]] != cut[edge[1]]:
#         # This edge is part of the cut, draw a dotted line
#         points = np.array([layout[edge[0]], layout[edge[1]]])
#         plt.plot(points[:, 0], points[:, 1], linestyle='--', color='blue')
plt.show()
