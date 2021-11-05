import networkx as nx
import numpy as np

def mf(config, G):
    k = config['k']
    A = np.array(nx.adjacency_matrix(G).todense())
    D = np.diag(np.sum(A, axis=1))
    I = np.identity(A.shape[0])
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    L = I - np.dot(D_inv_sqrt, A).dot(D_inv_sqrt)
    e, v = np.linalg.eig(L)
    sorted = np.argsort(e)
    ind = sorted[e[sorted] > 0][:k]
    out_e = e[ind]
    emb = v[ind]
    f = open(config['emb-path'], "w")
    for i in range(out_e.shape[0]):
        f.write("{}".format(out_e[i]))
        for e in emb[i]:
            f.write("{}".format(e))
        f.write("\n")
    f.close()
