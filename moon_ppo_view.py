import matplotlib.pyplot as plt
import networkx as nx

# Initialize a directed graph
G = nx.DiGraph()

# Define the layers and connections based on the provided network structure
# Input
G.add_node("Input", color="#cce5ff", layer_type="input")

# Policy network layers
policy_layers = [f"Policy L{i} (Linear+Tanh)" for i in range(1, 11)]
for i, layer in enumerate(policy_layers):
    G.add_node(layer, color="#ffd2cc", layer_type="policy")
    if i == 0:
        G.add_edge("Input", layer)
    else:
        G.add_edge(policy_layers[i - 1], layer)

# Policy Action Head
G.add_node("Action Head", color="#ffffcc", layer_type="action")
G.add_edge(policy_layers[-1], "Action Head")

# Value network layers
value_layers = [f"Value L{i} (Linear+Tanh)" for i in range(1, 11)]
for i, layer in enumerate(value_layers):
    G.add_node(layer, color="#ccffcc", layer_type="value")
    if i == 0:
        G.add_edge("Input", layer)
    else:
        G.add_edge(value_layers[i - 1], layer)

# Value Head
G.add_node("Value Head", color="#ffffcc", layer_type="value_head")
G.add_edge(value_layers[-1], "Value Head")

# Define positions for layers (x-axis: layer number, y-axis: network branch)
pos = {}
x_offset = 2
y_policy = 1.0
y_value = -1.0
pos["Input"] = (0, 0)

# Positioning for policy layers
for i, layer in enumerate(policy_layers):
    pos[layer] = (x_offset * (i + 1), y_policy)

pos["Action Head"] = (x_offset * (len(policy_layers) + 1), y_policy)

# Positioning for value layers
for i, layer in enumerate(value_layers):
    pos[layer] = (x_offset * (i + 1), y_value)

pos["Value Head"] = (x_offset * (len(value_layers) + 1), y_value)

# Extract colors for nodes
node_colors = [G.nodes[node].get('color', '#ffffff') for node in G.nodes]

# Draw the graph
plt.figure(figsize=(14, 6))
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000, font_size=9, edge_color="black", arrowsize=15)

# Add a title
plt.title("Policy-Value Network Architecture", fontsize=16)
plt.tight_layout()
plt.show()
