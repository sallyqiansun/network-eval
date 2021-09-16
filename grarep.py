import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import argparse
import networkx as nx

def parse_args():
    parser = argparse.ArgumentParser(description="Run GraRep")

    parser.add_argument('--input', nargs='?', default='examples/cora.csv', help='Input graph path')

    parser.add_argument('--output', nargs='?', default='embedding/cora-grarep.csv', help='Embeddings path')

    parser.add_argument('--format', nargs='?', default='edgelist', help='File format of input file')

    parser.add_argument('--iter', default=1, type=int, help="Number of epochs")

    parser.add_argument('--dimensions', type=int, default=16, help='Number of dimensions. Default is 16')

    parser.add_argument('--order', type=int, default=5, help='Approximation order. Default is 5.')

    parser.add_argument('--seed', type=int, default=42, help='Random seed. Default is 42.')

    parser.add_argument('--weighted', dest='weighted', action='store_true', help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true', help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()

def read_graph(args):
    if args.format == "edgelist":
        return read_graph_edgelist(args)

def read_graph_edgelist(args):
    if args.input[-3:] == "csv":
        # G = nx.parse_edgelist(args.input, delimiter=',', nodetype=int)
        data = open(args.input, "r")
        next(data, None)  # skip the first line in the input file
        Graphtype = nx.Graph()

        G = nx.parse_edgelist(data, delimiter=',', create_using=Graphtype,
                              nodetype=int, data=(('weight', float),))
    else:
        G = nx.read_edgelist(args.input, nodetype=int)

    if args.weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
    return G

def create_inverse_degree_matrix(edges):
    """
    Creating an inverse degree matrix from an edge list.
    :param edges: Edge list.
    :return D_1: Inverse degree matrix.
    """
    graph = nx.from_edgelist(edges)
    ind = range(len(graph.nodes()))
    degs = [1.0/graph.degree(node) for node in range(graph.number_of_nodes())]

    D_1 = sparse.coo_matrix((degs, (ind, ind)),
                            shape=(graph.number_of_nodes(),
                            graph.number_of_nodes()),
                            dtype=np.float32)

    return D_1

def normalize_adjacency(edges):
    """
    Method to calculate a sparse degree normalized adjacency matrix.
    :param edges: Edge list of graph.
    :return A: Normalized adjacency matrix.
    """
    D_1 = create_inverse_degree_matrix(edges)
    index_1 = [edge[0] for edge in edges] + [edge[1] for edge in edges]
    index_2 = [edge[1] for edge in edges] + [edge[0] for edge in edges]
    values = [1.0 for edge in edges] + [1.0 for edge in edges]
    A = sparse.coo_matrix((values, (index_1, index_2)),
                          shape=D_1.shape,
                          dtype=np.float32)
    A = A.dot(D_1)
    return A



def _setup_base_target_matrix(A):
    """
    Creating a base matrix to multiply.
    """
    values = [1.0 for i in range(A.shape[0])]
    indices = [i for i in range(A.shape[0])]
    A_hat = sparse.coo_matrix((values, (indices, indices)),
                                   shape=A.shape,
                                   dtype=np.float32)
    return A_hat

def _create_target_matrix(A, A_hat):
    """
    Creating a log transformed target matrix.
    :return target_matrix: Matrix to decompose with SVD.
    """
    A_hat = sparse.coo_matrix(A_hat.dot(A))
    scores = np.log(A_hat.data)-math.log(A.shape[0])
    rows = A_hat.row[scores < 0]
    cols = A_hat.col[scores < 0]
    scores = scores[scores < 0]
    target_matrix = sparse.coo_matrix((scores, (rows, cols)),
                                      shape=A.shape,
                                      dtype=np.float32)
    return target_matrix

def optimize(args, A, A_hat):
    """
    Learning an embedding.
    """
    print("\nOptimization started.\n")
    embeddings = []
    for step in tqdm(range(args.order)):
        target_matrix = _create_target_matrix(A, A_hat)

        svd = TruncatedSVD(n_components=args.dimensions,
                           n_iter=args.iter,
                           random_state=args.seed)

        svd.fit(target_matrix)
        embedding = svd.transform(target_matrix)
        embeddings.append(embedding)
    return embeddings

def save_embedding(args, A, A_hat):
    """
    Saving the embedding.
    """
    embeddings = optimize(args, A, A_hat)
    print("\nSaving embedding.\n")
    embeddings = np.concatenate(embeddings, axis=1)
    column_count = args.order*args.dimensions
    # columns = ["ID"] + ["x_" + str(col) for col in range(column_count)]
    # ids = np.array([i for i in range(A.shape[0])]).reshape(-1,1)
    # embeddings = np.concatenate([ids, embeddings], axis=1)
    # embeddings = pd.DataFrame(embeddings, columns=columns)
    embeddings = pd.DataFrame(embeddings)
    embeddings.to_csv(args.output, index=None)


def main(args):
    G = read_graph(args)
    A = normalize_adjacency(G.edges())
    A_hat = _setup_base_target_matrix(A)
    save_embedding(args, A, A_hat)

if __name__ == "__main__":
	args = parse_args()
	main(args)