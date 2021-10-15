import numpy
import sys

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict
from six import iteritems
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, zero_one_loss, log_loss
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer
import networkx as nx

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels


def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i,j,v in zip(cx.row, cx.col, cx.data):
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k,v in iteritems(G)}


def run(graph, config):
    embeddings_file = config['emb-path']

    # reference for implementation of reading from embeddings_file: https://github.com/xiangyue9607/BioNEV
    with open(embeddings_file) as f:
        f.readline().split()
        embeddings = {}
        for line in f:
            l = line.strip().split()
            node = l[0]
            embedding = l[1:]
            embedding = [float(i) for i in embedding]
            embedding = embedding / numpy.linalg.norm(embedding)
            numpy.nan_to_num(embedding, nan=0)
            embeddings[node] = list(embedding)
    f.close()

    # 2. Get labels from gpickle graph file
    labels_matrix = nx.adjacency_matrix(graph)
    mlb = MultiLabelBinarizer(range(labels_matrix.shape[1]))

    # Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
    features_matrix = numpy.asarray([embeddings[str(node+1)] for node in range(len(graph.nodes()))])

    # 2. Shuffle, to create train/test groups
    shuffles = []
    for x in range(config['num-shuffles']):
        shuffles.append(skshuffle(features_matrix, labels_matrix))

    # 3. to score each train/test group
    all_results = defaultdict(list)

    training_percents = config['train_percent']
    for train_percent in training_percents:
        for shuf in shuffles:

            X, y = shuf

            training_size = int(train_percent * X.shape[0])

            X_train = X[:training_size, :]
            y_train_ = y[:training_size]

            y_train = [[] for x in range(y_train_.shape[0])]

            cy = y_train_.tocoo()
            for i, j in zip(cy.row, cy.col):
                y_train[i].append(j)

            assert sum(len(l) for l in y_train) == y_train_.nnz

            X_test = X[training_size:, :]
            y_test_ = y[training_size:]

            y_test = [[] for _ in range(y_test_.shape[0])]

            cy = y_test_.tocoo()
            for i, j in zip(cy.row, cy.col):
                y_test[i].append(j)

            clf = TopKRanker(LogisticRegression())
            clf.fit(X_train, y_train_)

            # find out how many labels should be predicted
            top_k_list = [len(l) for l in y_test]
            preds = clf.predict(X_test, top_k_list)

            results = {}
            averages = ["micro", "macro"]
            for average in averages:
                results[average] = f1_score(mlb.fit_transform(y_test), mlb.fit_transform(preds), average=average)

            results['accuracy'] = accuracy_score(mlb.fit_transform(y_test), mlb.fit_transform(preds))
            results['zero_one_loss'] = zero_one_loss(mlb.fit_transform(y_test), mlb.fit_transform(preds))
            results['log_loss'] = log_loss(mlb.fit_transform(y_test), mlb.fit_transform(preds))

            all_results[train_percent].append(results)

    return all_results