from sklearn.cluster import KMeans
import numpy as np
import networkx as nx

def run(config, G):
    num_nodes = G.number_of_nodes()
    nodes = list(G.nodes())
    num_labels = len(list(nx.get_node_attributes(G,'label').values()))
    on_diag = int(num_nodes*0.5 + 1)
    off_diag = num_nodes - on_diag

    prob = np.full((num_labels, num_labels), off_diag/num_nodes)
    np.fill_diagonal(prob, on_diag/num_nodes)
    labels = np.random.randint(0, num_labels, num_nodes)

    mat = np.zeros((num_nodes, num_nodes), dtype=bool)
#     for i in range(num_nodes):
#         for j in range(i):
#             p = prob[labels[i], labels[j]]
#             if np.random.rand() <= p:
#                 mat[i][j] = 1
#     mat += mat.T

    mat = nx.adjacency_matrix(G).todense()

    d = np.sum(mat, axis=0)
    r = np.sqrt(np.mean(d))
    diag = np.diag(d)
    hes = (r**2-1)*np.identity(len(d)) - r*mat + diag

    pred = SpectralClustering(num_labels, hes)

    f = open(config['emb-path'], "w")
    for i in range(len(pred)):
        f.write("{} ".format(nodes[i]))
        f.write("{} ".format(pred[i]))
        f.write("\n")
    f.close()
    print("Output saved to {}. Note that this is not embeddings but predicted labels. ".format(config['emb-path']))


def SpectralClustering(num_labels, H):
    e, v = np.linalg.eig(H)
    ind = v.argsort()[:num_labels]
    out = v[:,ind]
    kmeans = KMeans(n_clusters=num_labels).fit(out)
    return kmeans.labels_