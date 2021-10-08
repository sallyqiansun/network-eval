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


def run(network_file, config):
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

    # 2. Load labels
    if config['format'] == "mat":
        mat = loadmat(network_file)
        A = mat[config['mat-variable-name']]
        graph = sparse2graph(A)
        labels_matrix = mat[config['mat-variable-name']]
        labels_count = labels_matrix.shape[1]
        mlb = MultiLabelBinarizer(range(labels_count))
    elif config['format'] == "adjlist":
        #TODO: finish this
        pass
    elif config['format'] == "edgelist":
        # TODO: finish this
        pass
    # Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
    features_matrix = numpy.asarray([embeddings[str(node)] for node in range(len(graph))])

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

    eva_fname = 'evaluation/'+ network_file[network_file.find('/'):network_file.find('.mat')] + '-' + embeddings_file[embeddings_file.find('-'):embeddings_file.find('.emd')] + '.txt'
    sys.stdout = open(eva_fname, "w")

    print('Results, using embeddings of dimensionality', X.shape[1])
    print('-------------------')
    for train_percent in sorted(all_results.keys()):
        print('Train percent:', train_percent)
        for index, result in enumerate(all_results[train_percent]):
            print('Shuffle #%d:   ' % (index + 1), result)
        avg_score = defaultdict(float)
        for score_dict in all_results[train_percent]:
            for metric, score in iteritems(score_dict):
                avg_score[metric] += score
        for metric in avg_score:
            avg_score[metric] /= len(all_results[train_percent])
        print('Average score:', dict(avg_score))
        print('-------------------')
    sys.stdout.close()