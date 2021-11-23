from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import networkx as nx

def visualize(config):
    path = config['emb-path']
    data_path = "data/" + config["data"] + ".gpickle"
    G = nx.read_gpickle(data_path)

    node_labels = dict()
    node_targets = nx.get_node_attributes(G, "label")
    labels = np.unique(list(node_targets.values()))
    label_map = {l: i for i, l in enumerate(labels)}

    with open(path) as f:
        f.readline().split()
        emb = {}
        for line in f:
            l = line.strip().split()
            node = l[0]
            node_labels[node] = node_targets[int(node)]
            embedding = l[1:]
            embedding = [float(i) for i in embedding]
            embedding = embedding / np.linalg.norm(embedding)
            np.nan_to_num(embedding, nan=0)
            emb[node] = list(embedding)
    f.close()

    embeddings = np.empty(shape=(len(emb), len(emb[list(emb.keys())[0]])))
    node_colours = []
    for i, k in enumerate(emb.keys()):
        embeddings[i, :] = emb[k]
        node_colours.append(label_map[node_targets[int(k)]])

    tsne = TSNE(n_components=2)
    node_embeddings_2d = tsne.fit_transform(embeddings)



    plt.figure(figsize=(10, 8))
    plt.scatter(
        node_embeddings_2d[:, 0],
        node_embeddings_2d[:, 1],
        c=node_colours,
        cmap="jet",
        alpha=0.7,
    )
    plt.savefig(config["fig-path"])


# import json
# config_file = open("config.json", "r")
# config = json.load(config_file)
# visualize(config)