from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np



def visualize(config):
    path = config['emb-path']
    with open(path) as f:
        f.readline().split()
        emb = {}
        i = 0
        for line in f:
            l = line.strip().split()
            node = l[0]
            embedding = l[1:]
            embedding = [float(i) for i in embedding]
            embedding = embedding / np.linalg.norm(embedding)
            np.nan_to_num(embedding, nan=0)
            emb[node] = list(embedding)
    f.close()

    embeddings = np.empty(shape=(len(emb), len(emb[list(emb.keys())[0]])))
    l_emb = list(emb)
    for i in range(len(emb)):
        embeddings[i, :] = emb[l_emb[i]]

    #TODO: reverse
    X_pca = PCA(n_components = 2).fit_transform(embeddings)
    kmeans = KMeans()
    kmeans.fit(X_pca)
    labels = kmeans.labels_
    plt.figure(figsize=(10, 10))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=100, cmap="prism", edgecolor="grey")
    for i, k in enumerate(list(emb.keys())):
        plt.annotate('', X_pca[i], horizontalalignment='center', verticalalignment='center',)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("PCA and K-means")
    plt.grid()
    plt.savefig(config['fig-path'])


# import json
# config_file = open("config.json", "r")
# config = json.load(config_file)
# visualize(config)