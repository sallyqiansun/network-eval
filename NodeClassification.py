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
    all_results = defaultdict(list)
    embeddings_file = config['emb-path']
    A = nx.adjacency_matrix(graph)

    shape_1 = 0
    # reference for implementation of reading from embeddings_file: https://github.com/xiangyue9607/BioNEV
    with open(embeddings_file) as f:
        f.readline().split()
        embeddings = {}
        shape_0 = 0
        for line in f:
            l = line.strip().split()
            node = l[0]
            embedding = l[1:]
            embedding = [float(i) for i in embedding]
            embedding = embedding / np.linalg.norm(embedding)
            np.nan_to_num(embedding, nan=0)
            embeddings[node] = list(embedding)
            shape_1 = len(embeddings[node])
            shape_0 += 1
    f.close()

    # 2. Get labels from gpickle graph file
    model = LabelPropagation()
    model.fit(nx.relabel.convert_node_labels_to_integers(graph, first_label=0, ordering='default'))
    cluster_membership = model.get_memberships()

    X = np.zeros((shape_0, shape_1))
    y = []
    i = 0
    for node in list(cluster_membership.keys()):
        node_ind = str(int(node) + 1)
        X[i][:] = embeddings[node_ind]
        y.append(cluster_membership[node])
        i += 1


    for train_pct in config['train_percent']:
        print("training percentage: ", train_pct)
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

        all_results[train_pct].extend([f1_micro, f1_macro, acc, zero_one])

    return all_results


