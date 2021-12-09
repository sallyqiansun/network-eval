import numpy as np
from karateclub import LabelPropagation
from collections import defaultdict
from six import iteritems
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, zero_one_loss, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
import networkx as nx

def run(graph, config):
    data_path = "data/" + config["data"] + ".gpickle"
    G = nx.read_gpickle(data_path)

    all_results = {}
    embeddings_file = config['emb-path']

    node_labels = {}
    node_targets = nx.get_node_attributes(G, "label")
    labels = np.unique(list(node_targets.values()))
    label_map = {l: i for i, l in enumerate(labels)}

    with open(embeddings_file) as f:
        f.readline().split()
        emb = {}
        for line in f:
            l = line.strip().split()
            node = l[0]
            if node in node_targets.keys():
                node_labels[node] = node_targets[node]
            else:
                continue
            embedding = l[1:]
            embedding = [float(i) for i in embedding]
            embedding = embedding / np.linalg.norm(embedding)
            np.nan_to_num(embedding, nan=0)
            emb[node] = list(embedding)
    f.close()

    print("Embedding loaded from {}.".format(config['emb-path']))

    X = np.empty(shape=(len(emb), len(emb[list(emb.keys())[0]])))
    y = np.empty(shape=len(emb))

    for i, k in enumerate(emb.keys()):
        X[i, :] = emb[k]
        y[i] = label_map[node_targets[k]]

    for train_pct in config['train_percent']:
        print("training percentage: ", train_pct)
        all_results[train_pct] = {}
        test_pct = 1 - float(train_pct)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = float(train_pct), test_size=test_pct, random_state=int(config['seed']))
        lr = LogisticRegression(random_state=int(config['seed'])).fit(X_train, y_train)
        y_hat = lr.predict(X_test)
        f1_micro = f1_score(y_test, y_hat, average="micro")
        f1_macro = f1_score(y_test, y_hat, average="macro")
        acc = accuracy_score(y_test, y_hat)
        zero_one = zero_one_loss(y_test, y_hat)
        print('micro F1: {:.4f}'.format(f1_micro))
        print('macro F1: {:.4f}'.format(f1_macro))
        print('accuracy score: {:.4f}'.format(acc))
        print('zero-one loss: {:.4f}'.format(zero_one))
        print()

        all_results[train_pct]["f1_micro"] = f1_micro
        all_results[train_pct]["f1_macro"] = f1_macro
        all_results[train_pct]["acc"] = acc
        all_results[train_pct]["zero_one_loss"] = zero_one

    predicted_on_all = lr.predict(X)
    true_label_on_all = y

    return all_results, X, predicted_on_all, true_label_on_all


