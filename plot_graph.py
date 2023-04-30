import networkx as nx
import matplotlib.pyplot as plt

def plot_colored_weighted_nodes(edge_list, node_weights, node_colors):
    # Create an empty graph
    G = nx.Graph()

    # Add edges to the graph from the edge list
    G.add_edges_from(edge_list)

    # Add node weights as attributes
    nx.set_node_attributes(G, node_weights, "weight")

    # Assign colors to nodes
    node_color_map = [node_colors[node] for node in G.nodes]

    # Draw the graph using NetworkX and Matplotlib
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=node_color_map, node_size=1000, font_size=12, font_weight="bold")
    
    # Draw node labels with weights
    node_labels = {node: f"{node} ({data['weight']})" for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_weight="bold")

    plt.show()

# Example usage
edge_list = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 5)]
node_weights = {1: 3, 2: 1, 3: 4, 4: 2, 5: 5}
node_colors = {1: "red", 2: "blue", 3: "green", 4: "yellow", 5: "purple"}

plot_colored_weighted_nodes(edge_list, node_weights, node_colors)






