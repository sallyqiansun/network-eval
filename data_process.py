import networkx as nx
from scipy.io import loadmat
from scipy.sparse import issparse
from networkx import Graph
import pandas as pd
import argparse
import csv

def save_graph(format, data, data_path, directed=False, weighted=False, variable_name="network", label_name="group"):
    G = read_graph(format, data_path, directed, weighted, variable_name, label_name)
    pickle_path = "data/" + data + ".gpickle"
    nx.write_gpickle(G, pickle_path)
    print("Data saved to", pickle_path)

def read_graph(format, data_path, directed=False, weighted=False, variable_name="network", label_name="group"):
    '''
    Reads the input network in networkx.
    '''
    if format == "edgelist":
        G = read_graph_edgelist(data_path, directed, weighted)
    elif format == "adjlist":
        G = read_graph_adjlist(data_path, directed, weighted)
    elif format == "mat":
        G = read_graph_mat(data_path, directed, weighted, variable_name, label_name)
    elif format == "csv":
        G = read_graph_csv(data_path, directed, weighted)
    else:
        raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat', 'csv" % format)

    return G


def read_graph_edgelist(input, directed, weighted):
    if weighted:
        G = nx.read_edgelist(input, nodetype=int, data=(('weight', float),))
    else:
        G = nx.read_edgelist(input, nodetype=int)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if directed == False:
        G = G.to_undirected()

    return G

def read_graph_adjlist(input, directed, weighted):
    G = nx.read_adjlist(input, nodetype=int)

    if directed == False:
        G = G.to_undirected()

    return G

def read_graph_mat(input, directed, weighted, variable_name):
    mat_varables = loadmat(input)
    x = mat_varables[variable_name]

    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            G.add_edge(i, j, weight=1)
    else:
        raise Exception("Dense matrices not yet supported.")

    if directed == False:
        G.to_undirected()

    return G

def read_graph_csv(input, directed, weighted):
    data = open(input, "r")
    if weighted:
        G = nx.read_edgelist(data, delimiter=',', nodetype=int, data=(('weight', float),))
    else:
        G = nx.read_edgelist(data, delimiter=',', nodetype=int)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if directed == False:
        G.to_undirected()

    return G


# save_graph('edgelist', 'karate', 'examples/karate.edgelist', directed=False, weighted=False)
data_path = "data/" + 'cora' + ".gpickle"
G = nx.read_gpickle(data_path)
print(G.edges())
# data= pd.read_csv('data/cora/group-edges.csv', header=None)
# print(data)


