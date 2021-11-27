from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import networkx as nx
from sklearn.manifold import TSNE

def visualize(result, config):
    X = result[1]
    X = TSNE(n_components=2).fit_transform(X)
    predicted_on_all = result[2]
    true_label_on_all = result[3]

    plt.rcParams["figure.figsize"] = (40,10)
    plt.subplot(1, 3, 1)
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=true_label_on_all,
        cmap="jet",
        alpha=0.7,
    )
    plt.title('{} embedding with true labels'.format(config["method"]))

    plt.subplot(1, 3, 2)
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=predicted_on_all,
        cmap="jet",
        alpha=0.7,
    )
    plt.title('{} embedding with predicted labels'.format(config["method"]))





    w_b = [1 if x == y else 0 for x, y, in zip(result[2], result[3])]

    plt.subplot(1, 3, 3)
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=w_b,
        cmap="jet",
        alpha=0.7,
    )
    plt.title('{} difference'.format(config["method"]))


    plt.savefig(config["fig-path"])


# import task
# import json
# config_file = open("config.json", "r")
# config = json.load(config_file)
# data_path = "data/" + config["data"] + ".gpickle"
# G = nx.read_gpickle(data_path)
# result = task.train(G, config)
# visualize(result, config)