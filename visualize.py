from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np



def visualize(path):
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

    embeddings = np.empty(shape=(len(emb), 100))
    l_emb = list(emb)
    for i in range(len(emb)):
        embeddings[i, :] = emb[l_emb[i]]

    emb_pca = PCA().fit_transform(embeddings)
    emb_tsne = TSNE().fit_transform(embeddings)


    fig, ax = plt.subplots()
    ax.scatter(emb_tsne[:, 0], emb_tsne[:, 1], s=3)
    for x, y, node in zip(emb_tsne[:, 0], emb_tsne[:, 1], list(emb)):
        ax.annotate('', xy=(x, y), size=8)
    fig.suptitle('node embeddings', fontsize=10)
    fig.set_size_inches(10, 10)
    plt.show()

