import numpy as np
from collections import namedtuple
from scipy.spatial import Delaunay
import unionfind
import random
import networkx as nx
import matplotlib.pyplot as plt

vertices = [
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1),
    (0.5, 0.5)
]

cells = [
    [0, 1, 4],
    [1, 2, 4],
    [2, 3, 4],
    [3, 0, 4]
]

class mesh:

    def __init__(self,vertices,cells):
        self.vertices = vertices
        self.cells = cells
# Define a mesh as a named tuple with two fields: vertices and cells
Mesh = namedtuple('Mesh', ['vertices', 'cells'])
def sample_function(mesh):
    # Assign random integer values to each vertex of the mesh
    f = {}
    for vertex in mesh.vertices:
        f[vertex] = random.randint(0, 9)
    return f
# Define a function to compute the intersection of a face with a given value
def get_intersection(face, val,f):
    for e in face.edges:
        v1, v2 = e.vertices
        if f[v1] < val <= f[v2] or f[v2] < val <= f[v1]:
            t = (val - f[v1]) / (f[v2] - f[v1])
            return np.array(v1) + t * (np.array(v2) - np.array(v1))
    return None

# Define a function to check if two fragments are adjacent
def are_adjacent(fragment_i, fragment_j):
    for vertex_i in fragment_i:
        for vertex_j in fragment_j:
            if np.allclose(vertex_i, vertex_j):
                return True
    return False

# Define a function to check if two fragments have the same values of f
def same_values(fragment_i, fragment_j, f):
    values_i = sorted(f[vertex] for vertex in fragment_i)
    values_j = sorted(f[vertex] for vertex in fragment_j)
    return values_i == values_j

def compute_jcn(mesh, f):
    # Phase I: Create Fragments
    fragments = []
    for cell in mesh.cells:
        min_val = min(f[v] for v in cell.vertices)
        max_val = max(f[v] for v in cell.vertices)
        for j in range(min_val, max_val):
            fragment = []
            for face in cell.faces:
                if min(f[v] for v in face.vertices) < j and j <= max(f[v] for v in face.vertices):
                    intersection = get_intersection(face, j)
                    if intersection:
                        fragment.append(intersection)
            fragments.append(fragment)
    
    # Phase II: Compute Joint Graph
    G = {}
    for i, fragment_i in enumerate(fragments):
        for j, fragment_j in enumerate(fragments[i+1:], start=i+1):
            if are_adjacent(fragment_i, fragment_j):
                G.setdefault(i, set()).add(j)
                G.setdefault(j, set()).add(i)
    
    # Phase III: Compute Nodes
    U = unionfind.unionfind(len(fragments))
    for i, fragment_i in enumerate(fragments):
        for j, fragment_j in enumerate(fragments[i+1:], start=i+1):
            if same_values(fragment_i, fragment_j, f):
                U.union(i, j)
    
    # Phase IV: Compute Edges
    JCN = nx.Graph()
    for i, fragment_i in enumerate(fragments):
        root_i = U.find(i)
        if root_i == i:
            JCN.add_node(i)
        for j in G.get(i, []):
            root_j = U.find(j)
            if root_i != root_j:
                JCN.add_edge(root_i, root_j)
    
    return JCN

# Define a mesh with vertices and cells

JCN = compute_jcn(Mesh, sample_function(Mesh))

G = nx.Graph()
for node in JCN.nodes:
    G.add_node(node)
for edge in JCN.edges:
    G.add_edge(edge[0], edge[1])

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True)
plt.show()
