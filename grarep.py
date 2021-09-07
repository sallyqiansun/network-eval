# adapted from https://github.com/DimBer/GraRep-Python3 Dimitris Berberidis

from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from numpy import loadtxt
import numpy as np
import sys
import argparse
import math

#TODO: reformat this code

def parse_input():
    parser = argparse.ArgumentParser(description='Input embedding method and graph name.')
    parser.add_argument('-g', '--graph_filename', metavar='', type=str, default='HomoSapiens.adj',
                        help='Path to graph edgelist')
    parser.add_argument('-e', '--embedding_filename', metavar='', type=str, default='emb/HomoSapiens.emb',
                        help='Path to file containing embeddings')
    parser.add_argument('-d', '--dimension', metavar='', type=int, default=100, help='Embedding dimension')
    parser.add_argument('--K', metavar='', type=int, default=5, help='Number of transition steps')
    parser.add_argument('-b', '--beta', metavar='', type=float, default=1.0, help='Bias parameter')
    parser.add_argument('--directed', metavar='', type=bool, default=False, help='Set true to treat graph as directed')
    args = parser.parse_args()
    return args.graph_filename, args.embedding_filename, args.dimension, args.K, args.beta, args.directed




def get_representations(A, K, beta):
    # Extract list of similarity matrices to be factorized

    print('Computing representations')
    A_temp = []
    X_rep = []

    A_o = csr_matrix(A)
    A_prev = csr_matrix(A)
    A_temp.append(A_o.todense())
    for k in range(K - 1):
        A_last = csr_matrix(A_prev.dot(A_o))
        A_temp.append(A_last.todense())
        A_prev = A_last

    k = 1
    for A_k in A_temp:
        print('step: ', k, '/', K)
        k += 1
        A_k = np.array(A_k)
        g = A_k.sum(axis=0)
        g = np.array(np.power(g * beta / float(len(g)), -1))
        B = np.log(A_k.dot(np.diag(np.reshape(g, [len(g)]))).clip(min=1.0))
        X_rep.append(B)

    return X_rep


def get_embeddings(X_rep, N, dimension, K):
    # Factorize (SVD) and concatenate similarity matrices

    print('Extracting embeddings..')
    E = np.ndarray((N, dimension))

    width = dimension // K

    mod = dimension - width * K

    print('step:  1/', K)
    U, S, V = svds(X_rep[0], width + mod)
    E[:, :width + mod] = U @ np.power(np.diag(S), 0.5)

    for k in range(K - 1):
        print('step: ', k + 2, '/', K)
        U, S, V = svds(X_rep[k + 1], width + mod)
        E[:, width * (k + 1) + mod: width * (k + 2) + mod] = U @ np.power(np.diag(S), 0.5)

    return E


def main():
    graph_filename, embedding_filename, dimension, K, beta, directed = parse_input()

    A = get_graph(graph_filename, directed)

    X_rep = get_representations(A, K, beta)

    E = get_embeddings(X_rep, A.get_shape()[0], dimension, K)

    np.savetxt(embedding_filename, E, delimiter=' ')

    print('Finished')


if __name__ == '__main__':
    main()