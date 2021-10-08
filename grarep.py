import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import argparse
import networkx as nx


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

def optimize(A, A_hat, config):
    """
    Learning an embedding.
    """
    print("\nOptimization started.\n")
    embeddings = []
    for step in tqdm(range(config['order'])):
        target_matrix = _create_target_matrix(A, A_hat)

        svd = TruncatedSVD(n_components=config['dimensions'],
                           n_iter=config['iter'],
                           random_state=config['seed'])

        svd.fit(target_matrix)
        embedding = svd.transform(target_matrix)
        embeddings.append(embedding)
    return embeddings

def save_embedding(config, A, A_hat):
    """
    Saving the embedding.
    """
    embeddings = optimize(A, A_hat, config)
    print("\nSaving embedding.\n")
    embeddings = np.concatenate(embeddings, axis=1)
    column_count = config['order'] * config['dimensions']
    # columns = ["ID"] + ["x_" + str(col) for col in range(column_count)]
    # ids = np.array([i for i in range(A.shape[0])]).reshape(-1,1)
    # embeddings = np.concatenate([ids, embeddings], axis=1)
    # embeddings = pd.DataFrame(embeddings, columns=columns)
    embeddings = pd.DataFrame(embeddings)
    embeddings.to_csv(config['emb-path'], index=None)


def simulate_and_embed(config, G):
    A = normalize_adjacency(G.edges())
    A_hat = _setup_base_target_matrix(A)
    save_embedding(config, A, A_hat)