from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import networkx as nx

def visualize(result, config):
    X = result[1]
    predicted_on_all = result[2]
    true_label_on_all = result[3]

    plt.rcParams["figure.figsize"] = (40,10)
    plt.subplot(1, 2, 1)
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=true_label_on_all,
        cmap="jet",
        alpha=0.7,
    )
    plt.title('{} embedding with true labels'.format(config["method"]))

    plt.subplot(1, 2, 2)
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=predicted_on_all,
        cmap="jet",
        alpha=0.7,
    )
    plt.title('{} embedding with predicted labels'.format(config["method"]))

    plt.savefig(config["fig-path"])


# import json
# config_file = open("config.json", "r")
# config = json.load(config_file)
# visualize(config)