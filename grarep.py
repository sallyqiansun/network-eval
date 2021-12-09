import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import argparse
import networkx as nx



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
    embeddings = []
    for i in (range(config['order'])):
        print (str(i+1), '/', str(config['order']))
        target_matrix = _create_target_matrix(A, A_hat)

        svd = TruncatedSVD(n_components=config['dimensions'],
                           n_iter=config['iter'],
                           random_state=config['seed'])

        svd.fit(target_matrix)
        embedding = svd.transform(target_matrix)
        embeddings.append(embedding)
    return embeddings

def save_embedding(config, A, A_hat, G):
    """
    Saving the embedding.
    """
    nodes = sorted(G.nodes())
    # save original node_ids using dictionary
    node_dict = {}
    i = 0
    while i < len(nodes):
        node_dict[i] = nodes[i]
        i += 1

    embeddings = optimize(A, A_hat, config)
    emb = np.concatenate(embeddings, axis=1)

    f = open(config['emb-path'], "w")
    f.write("{} {}\n".format(emb.shape[0], emb.shape[1]))
    emb = emb.tolist()
    for i in range(len(emb)):
        f.write("{} ".format(node_dict[i]))
        for e in emb[i]:
            f.write("{} ".format(e))
        f.write("\n")
    f.close()
    print("Embedding saved to {}.".format(config['emb-path']))


def run(config, G):
    A = nx.adjacency_matrix(G).todense()
    A_hat = _setup_base_target_matrix(A)
    save_embedding(config, A, A_hat, G)