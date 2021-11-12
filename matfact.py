import networkx as nx
import numpy as np

def run(config, G):
    k = config['dimensions']
    A = np.array(nx.adjacency_matrix(G).todense())
    D = np.diag(np.sum(A, axis=1))
    I = np.identity(A.shape[0])
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    L = I - np.dot(D_inv_sqrt, A).dot(D_inv_sqrt)
    e, v = np.linalg.eig(L)
    sorted = np.argsort(e)
    ind = sorted[e[sorted] != 0][:]
    emb = v[ind]
    f = open(config['emb-path'], "w")
    f.write("{} {}\n".format(emb.shape[0], emb.shape[1]))
    for i in ind:
        f.write("{} ".format(i+1))
        for e in emb[i-1]:
            f.write("{} ".format(e))
        f.write("\n")
    f.close()
