import networkx as nx
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import node2vec as N2V
import random as rand

def run(network_file, config):
    # 0. Files
    embeddings_file = config['emb-path']

    # 1. Load Embeddings
    # reference for implementation of reading from embeddings_file: https://github.com/xiangyue9607/BioNEV
    with open(embeddings_file) as f:
        f.readline().split()
        embeddings = {}
        for line in f:
            l = line.strip().split()
            node = l[0]
            embedding = l[1:]
            embedding = [float(i) for i in embedding]
            embedding = embedding / np.linalg.norm(embedding)
            np.nan_to_num(embedding, nan=0)
            embeddings[node] = list(embedding)
    f.close()

    # 2. Read graph
    G = N2V.read_graph(config['format'])


    # 3. Set positive and negative training sets
    for train_percent in config['train_percent']:
        labeled = generate_pos_neg_links(G)
        # TODO: finish this training function

    # 4. Evaluate


# codes for train-test split of edges are adapted from https://github.com/adocherty/node2vec_linkprediction
def generate_pos_neg_links(G):
    """
    Select random existing edges in the graph to be postive links,
    and random non-edges to be negative links. proportion is 1:1
    Modify graph by removing the postive links.
    This function returns the selected links, and their corresponding labels
    """
    # Select n edges at random (positive samples)
    n_edges = len(G.edges())
    n_nodes = len(G.nodes())
    npos = int(0.5 * n_edges)
    nneg = n_edges - npos

    n_neighbors = [len(G.neighbors(v)) for v in G.nodes_iter()]
    n_non_edges = n_nodes - 1 - np.array(n_neighbors)

    non_edges = [e for e in nx.non_edges(G)]
    print("Finding %d of %d non-edges" % (nneg, len(non_edges)))

    # Select m pairs of non-edges (negative samples)
    rnd_inx = rand.choice(len(non_edges), nneg, replace=False)
    neg_edge_list = [non_edges[ii] for ii in rnd_inx]

    if len(neg_edge_list) < nneg:
        raise RuntimeWarning(
            "Only %d negative edges found" % (len(neg_edge_list))
        )

    print("Finding %d positive edges of %d total edges" % (npos, n_edges))

    # Find positive edges, and remove them.
    edges = G.edges()
    pos_edge_list = []
    n_count = 0
    n_ignored_count = 0
    rnd_inx = rand.permutation(n_edges)
    for eii in rnd_inx:
        edge = edges[eii]

        # Remove edge from graph
        data = G[edge[0]][edge[1]]
        G.remove_edge(*edge)

        # Check if graph is still connected
        reachable_from_v1 = nx.connected._plain_bfs(G, edge[0])
        if edge[1] not in reachable_from_v1:
            G.add_edge(*edge, **data)
            n_ignored_count += 1
        else:
            pos_edge_list.append(edge)
            print("Found: %d    " % (n_count), end="\r")
            n_count += 1

        # Exit if we've found npos nodes or we have gone through the whole list
        if n_count >= npos:
            break

    if len(pos_edge_list) < npos:
        raise RuntimeWarning("Only %d positive edges found." % (n_count))

    selected = pos_edge_list + neg_edge_list
    labels = np.zeros(len(selected))
    labels[:len(pos_edge_list)] = 1
    return selected, labels

