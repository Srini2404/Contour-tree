import numpy as np
from collections import namedtuple
from scipy.spatial import Delaunay
# from . import unionfind
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import sys
from queue import Queue


class Node:
    def __init__(self,val,flg,dir,cell_no):
        self.val = val
        self.flg = flg
        self.dir = dir
        self.cell_no = cell_no

    def __str__(self):
        return str(self.val)+" "+str(self.flg)+" "+str(self.dir)+" "+str(self.cell_no)


class cell_Graph:
    global f
    global n
    def __init__(self,cellEdges,tpl,cell_no,p): # tpl is the top left index for the Cell
        self.ListUp = [0,3,4]
        self.ListDown = [1,2,4]
        self.Indices = [tpl,tpl+1,tpl+n+1,tpl,tpl+n,tpl+n+1]
        self.Index_in_nodes = []
        counter = -1
        self.Nodes = []

        self.cellEdges = cellEdges
        self.diagonalNodesUp = []
        self.diagonalNodesDown = []
        # Order in which we are adding the vertices - top left,top right,bottom right,top left,bottom left,bottom right
        # Zero is a dummy value we may need to change it later.
        
        for i in range(3):
            counter+=1
            a1 = Node(f[self.Indices[counter]],0,'U',cell_no) 
            self.Nodes.append(a1)
            self.Index_in_nodes.append(len(self.Nodes)-1)

            if (counter == 2 or counter == 0):
                self.diagonalNodesUp.append(p+len(self.Nodes) -1)
            # for/ j in range(i+1,3):
            for j in cellEdges[self.ListUp[i]]:
                prev = self.Nodes[-1].val
                if(prev <j):
                    a1 = Node(j,1,'U',cell_no)
                    a2 = Node(j,2,'U',cell_no)
                else:
                    a1 = Node(j,2,'U',cell_no)
                    a2 = Node(j,1,'U',cell_no)
                self.Nodes.append(a1)
                self.Nodes.append(a2)

                if counter == 2 :
                    self.diagonalNodesUp.append(p+len(self.Nodes)-1)
                    self.diagonalNodesUp.append(p+len(self.Nodes)-2)

            
            
        for i in range(3):
            counter+=1
            a1 = Node(f[self.Indices[counter]],0,'D',cell_no) 
            self.Nodes.append(a1)
            self.Index_in_nodes.append(len(self.Nodes)-1)
            if (counter == 3 or counter == 5):
                self.diagonalNodesDown.append(p+len(self.Nodes) -1)

            for j in cellEdges[self.ListDown[i]]:
                prev = self.Nodes[-1].val
                if(prev <j):
                    a1 = Node(j,1,'D',cell_no)
                    a2 = Node(j,2,'D',cell_no)
                else:
                    a1 = Node(j,2,'D',cell_no)
                    a2 = Node(j,1,'D',cell_no)
                self.Nodes.append(a1)
                self.Nodes.append(a2)

                if counter == 5 :
                    self.diagonalNodesUp.append(p+len(self.Nodes)-1)
                    self.diagonalNodesUp.append(p+len(self.Nodes)-2)

    

    def generate_Graph(self,p):
        index = 0
        curr_index = 1

        Edges = []

        while index < len(self.Nodes):
            if (curr_index < 6):
                if (index < self.Index_in_nodes[curr_index]):
                    if (index == self.Index_in_nodes[curr_index] - 1 and curr_index == 3):
                        Edges.append((p+index, p))
                        index+=2
                    
                    else:
                        Edges.append((p+index, p+index+1))
                        index+=2
                else:
                    index = self.Index_in_nodes[curr_index]
                    curr_index+=1
            else:
                if (index == len(self.Nodes)-1):
                    Edges.append((p+index, p+self.Index_in_nodes[3]))
                    index+=2
                else:
                    Edges.append((p+index, p+index+1))
                    index+=2
        
        for i in range(5):
            if len(cellEdges[i]) == 0:
                if (i==0):
                    if (p+self.Index_in_nodes[0],p+self.Index_in_nodes[1]) not in Edges:
                        Edges.append((p+self.Index_in_nodes[0],p+self.Index_in_nodes[1]))
                elif (i==1):
                    if (p+self.Index_in_nodes[3],p+self.Index_in_nodes[4]) not in Edges:
                        Edges.append((p+self.Index_in_nodes[3],p+self.Index_in_nodes[4]))
                elif (i==2):
                    if (p+self.Index_in_nodes[4],p+self.Index_in_nodes[5]) not in Edges:
                        Edges.append((p+self.Index_in_nodes[4],p+self.Index_in_nodes[5]))
                elif (i==3):
                    if (p+self.Index_in_nodes[1],p+self.Index_in_nodes[2]) not in Edges:
                        Edges.append((p+self.Index_in_nodes[1],p+self.Index_in_nodes[2]))
                elif (i==4):
                    if (p+self.Index_in_nodes[2],p+self.Index_in_nodes[0]) not in Edges:
                        Edges.append((p+self.Index_in_nodes[2],p+self.Index_in_nodes[0]))
                    if (p+self.Index_in_nodes[5],p+self.Index_in_nodes[3]) not in Edges:
                        Edges.append((p+self.Index_in_nodes[5],p+self.Index_in_nodes[3]))

        for i in range(0,len(self.Nodes)-1):
            for j in range(i+1, len(self.Nodes)):
                if (self.Nodes[i].val == self.Nodes[j].val and self.Nodes[i].flg == self.Nodes[j].flg and self.Nodes[i].dir == self.Nodes[j].dir):
                    if (p+i,p+j) not in Edges and (p+j,p+i) not in Edges:
                        Edges.append((p+i,p+j))
        
        return Edges
    
    def getDiagonals(self):
        return (self.diagonalNodesUp, self.diagonalNodesDown)


def sample_function(vertices):
    f = {}
    for i in range (0,len(vertices)):
        f[i] = random.randint(1,9)
    return f

vertices = [
    (0, 1),
    (0.5, 1),
    (1, 1),
    (0, 0.5),
    (0.5, 0.5),
    (1, 0.5),
    (0, 0),
    (0.5, 0),
    (1, 0)
]

n = int(math.sqrt(len(vertices)))
f = {}
# f = sample_function(vertices)
f[0] = 1
f[1] = 2
f[2] = 3
f[3] = 2
f[4] = 1
f[5] = 2
f[6] = 3
f[7] = 2
f[8] = 1
print(f)

min_val = min(f.values())
max_val = max(f.values())

diff = 1
breakpoints = []

i = min_val - 0.5 # we are doing i+=diff at the start so to get the correct fragment values.
while (i<max_val):
    i+=diff
    breakpoints.append(i)

temp = len(breakpoints)-1
del breakpoints[temp]

def bin_srch_left(val):
    global breakpoints
    l=0
    r=len(breakpoints)-1

    while (l<r):

        m = l + (r-l)//2
        if breakpoints[m] == val:
            return m-1
        elif breakpoints[m] > val:
            r=m-1
        else:
            if (breakpoints[m+1] < val):
                l=m+1
            else:
                return m
    if (breakpoints[l] < val):
        return l
    else:
        return -1

def bin_srch_right(val):
    global breakpoints
    l=0
    r=len(breakpoints)-1

    while (l<r):
        m = l + (r-l)//2
        if breakpoints[m] == val:
            return m+1
        elif breakpoints[m] > val:
            r=m
        else:
            l=m+1
    
    if (breakpoints[l] > val):
        return l
    else:
        return -1

def edge_brk_points(idx1,idx2):
    global breakpoints
    stidx = bin_srch_right(idx1)
    edidx = bin_srch_left(idx2)
    if (stidx == -1 or edidx == -1):
        return []
    return breakpoints[stidx:edidx+1]

def get_cell_fragments(idx,n):
    global f
    cellEdges = []
    l1 = edge_brk_points(min(f[idx],f[idx+1]),max(f[idx],f[idx+1]))
    if f[idx]>f[idx+1]:
        l1.reverse()  
    cellEdges.append(l1)
    l1 = edge_brk_points(min(f[idx],f[idx+n]),max(f[idx],f[idx+n]))
    if f[idx] > f[idx+n]:
        l1.reverse()
    cellEdges.append(l1)
    l1 = edge_brk_points(min(f[idx+n],f[idx+n+1]),max(f[idx+n],f[idx+n+1]))
    if f[idx+n] > f[idx+n+1]:
        l1.reverse()
    cellEdges.append(l1)
    l1 = edge_brk_points(min(f[idx+n+1],f[idx+1]),max(f[idx+n+1],f[idx+1]))
    if f[idx+n+1] < f[idx+1]:
        l1.reverse()
    cellEdges.append(l1)
    l1 = edge_brk_points(min(f[idx+n+1],f[idx]),max(f[idx+n+1],f[idx]))
    if f[idx+n+1] > f[idx]:
        l1.reverse()
    cellEdges.append(l1)
    return cellEdges


def adjlist_edglist(EdgeList):
    global adjlist
    for v in EdgeList:
        v1,v2 = v
        adjlist[v1].append(v2)
        adjlist[v2].append(v1)

Edges_global = []
Nodes_global = []
Universal_edges = []
p=0
adjlist = []

for i in range(n-1):
    for j in range (n-1):
        idx = n*i+j
        # print("In")
        cellEdges = get_cell_fragments(idx,n)
        print(cellEdges)        
        # print("Done")
        G = cell_Graph(cellEdges,idx,(n-1)*i+j,p)
        EdgeList = G.generate_Graph(p)
        NodesList = G.Nodes
        Diagonal_in_cell = G.getDiagonals()
        Universal_edges.append(Diagonal_in_cell)
        Edges_global.extend(EdgeList)
        Nodes_global.extend(NodesList)
        p+=len(NodesList)

for i in range(p):
    adjlist.append([])

adjlist_edglist(Edges_global)
visited = [0 for i in range(p)]
components = []


def bfs(adjlist,source):
    global visited
    global components

    q = Queue()
    l=[]
    q.put(source)
    while q.empty()==False :
        node = q.get()
        for u in adjlist[node]:
            if visited[u] == 0:
                q.put(u)
                visited[u] = 1
                l.append(u)
    
    components.append(l)

for i in range(p):
    if visited[i] == False :
        bfs(adjlist, i)

final_nodes = []
for l in components:
    l1 = [Nodes_global[u].val for u in l]
    max_l = math.floor(max(l1))
    final_nodes.append(max_l)

Edges_Tree = []

def connect_nodes():
    global components
    global Edges_Tree
    global Universal_edges
    for i in range(len(components)-1):
        for j in range(i+1, len(components)):
            comp1 = components[i]
            comp2 = components[j]
            flg = 0
            for k in comp1:
                for l in comp2:
                    if Nodes_global[k].cell_no == Nodes_global[l].cell_no:
                        if Nodes_global[k].val == Nodes_global[l].val:
                            if ( Nodes_global[k].dir == Nodes_global[l].dir):
                                flg = 1
                                Edges_Tree.append([min(i,j),max(i,j)])
                                break
                            elif (Nodes_global[k].flg == Nodes_global[l].flg):
                                count1 = count2 = 0
                                for p in comp1:
                                    if p in Universal_edges[Nodes_global[p].cell_no][0] or p in Universal_edges[Nodes_global[p].cell_no][1]:
                                        count1+=1
                                for p in comp1:
                                    if p in Universal_edges[Nodes_global[p].cell_no][0] or p in Universal_edges[Nodes_global[p].cell_no][1]:
                                        count2+=1
                                
                                if (count2 == count1 and count1 == 2):
                                    flg = 1
                                    Edges_Tree.append([min(i,j),max(i,j)])
                                    break
                    else:   
                        if Nodes_global[k].cell_no - (n-1) == Nodes_global[l].cell_no:
                            if Nodes_global[k].val == Nodes_global[l].val:
                                if(Nodes_global[k].flg == Nodes_global[l].flg and Nodes_global[k].dir == "U" and Nodes_global[l].dir == "D"):
                                    flg = 1
                                    Edges_Tree.append([min(i,j),max(i,j)])
                                    break

                        elif (Nodes_global[k].cell_no -1 == Nodes_global[l].cell_no and Nodes_global[k].cell_no%(n-1)):
                            if Nodes_global[k].val == Nodes_global[l].val:
                                if(Nodes_global[k].flg == Nodes_global[l].flg and Nodes_global[k].dir == "D" and Nodes_global[l].dir == "U"):
                                    flg = 1
                                    Edges_Tree.append([min(i,j),max(i,j)])
                                    break

                        elif (Nodes_global[k].cell_no +1 == Nodes_global[l].cell_no and Nodes_global[l].cell_no%(n-1)):
                            if Nodes_global[k].val == Nodes_global[l].val:
                                if(Nodes_global[k].flg == Nodes_global[l].flg and Nodes_global[k].dir == "U" and Nodes_global[l].dir == "D"):
                                    flg = 1
                                    Edges_Tree.append([min(i,j),max(i,j)])
                                    break
                        elif (Nodes_global[k].cell_no +(n-1) == Nodes_global[l].cell_no):
                            if Nodes_global[k].val == Nodes_global[l].val:
                                if(Nodes_global[k].flg == Nodes_global[l].flg and Nodes_global[k].dir == "D" and Nodes_global[l].dir == "U"):
                                    flg = 1
                                    Edges_Tree.append([min(i,j),max(i,j)])
                                    break
                if flg == 1:
                    break


connect_nodes()
print(Edges_Tree)

# print(final_nodes)

# print(len(components))
for i in Edges_Tree:
    v1,v2 = i[0],i[1]
    if final_nodes[v1]==final_nodes[v2]:
        for idx in range(len(Edges_Tree)):
            if Edges_Tree[idx][0] == v2:
                Edges_Tree[idx][0] = v1
            
            if Edges_Tree[idx][1] == v2:
                Edges_Tree[idx][1] = v1


setEdges = set()

for i in Edges_Tree:
    if (i[0] != i[1]):
        tup = tuple(i)
        setEdges.add(tup)

edge_list = list(setEdges)

visit = [0 for i in range(len(final_nodes))]

for e in edge_list:
    u,v = e
    visit[u] = 1
    visit[v] = 1


def plot_colored_weighted_nodes(edge_list, node_weights):
    # Create an empty graph
    G = nx.Graph()

    # Add edges to the graph from the edge list
    G.add_edges_from(edge_list)

    # Add node weights as attributes
    nx.set_node_attributes(G, node_weights, "weight")

    # Assign colors to nodes

    # Draw the graph using NetworkX and Matplotlib
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=False, node_size=1000, font_size=12, font_weight="bold")
    
    # Draw node labels with weights
    node_labels = {node: f"{data['weight']}" for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_weight="bold")

    plt.show()

# Example usage

node_weights = {}

for i in range(len(visit)):
    if visit[i] == 1:
        node_weights[i] = final_nodes[i]

plot_colored_weighted_nodes(edge_list, node_weights)