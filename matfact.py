import networkx as nx
import numpy as np

def run(config, G):
    nodes = sorted(G.nodes())
    #save original node_ids using dictionary
    node_dict = {}
    i = 0
    while i < len(nodes):
        node_dict[i] = nodes[i]
        i += 1

    k = config['dimensions']
    A = np.array(nx.adjacency_matrix(G).todense())
    D = np.diag(np.sum(A, axis=1))
    I = np.identity(A.shape[0])
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    L = I - np.matmul(np.matmul(D_inv_sqrt, A), D_inv_sqrt)
    e, v = np.linalg.eig(L)
    e = np.real(e)
    v = np.real(v)
    ind = np.argsort(e)[:k]
    # ind = sorted[e[sorted] != 0][:]
    emb = v[ind].T
    f = open(config['emb-path'], "w")
    f.write("{} {}\n".format(emb.shape[0], emb.shape[1]))
    emb = emb.tolist()
    for i in range (len(emb)):
        f.write("{} ".format(node_dict[i]))
        for e in emb[i]:
            f.write("{} ".format(e))
        f.write("\n")
    f.close()
    print("Embedding saved to {}.".format(config['emb-path']))
