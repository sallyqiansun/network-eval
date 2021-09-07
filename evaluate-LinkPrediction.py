import sys
import numpy as np
from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec
from networkx import Graph
from scipy.sparse import issparse
from scipy.io import loadmat

def node2vec_embedding(args, graph):
    rw = BiasedRandomWalk(graph)
    walks = rw.run(graph.nodes(), n=args.num_walks, length=args.walk_length, p=args.p, q=args.q)
    # print(f"Number of random walks for '{name}': {len(walks)}")

    model = Word2Vec(
        walks,
        vector_size=args.representation_size,
        window=args.window_size,
        min_count=0,
        sg=1,
        workers=args.workers,
        epochs=args.iter,
    )

    def get_embedding(u):
        return model.wv[u]

    return get_embedding

def operator_hadamard(u, v):
    return u * v

def operator_l1(u, v):
    return np.abs(u - v)

def operator_l2(u, v):
    return (u - v) ** 2

def operator_avg(u, v):
    return (u + v) / 2.0


# 1. link embeddings
def link_examples_to_features(link_examples, transform_node, binary_operator):
    return [
        binary_operator(transform_node(src), transform_node(dst))
        for src, dst in link_examples
    ]

# 2. training classifier
def train_link_prediction_model(link_examples, link_labels, get_embedding, binary_operator):
    clf = link_prediction_classifier()
    link_features = link_examples_to_features(
        link_examples, get_embedding, binary_operator
    )
    clf.fit(link_features, link_labels)
    return clf

# 3. classifier
def link_prediction_classifier(max_iter=2000):
    lr_clf = LogisticRegressionCV(Cs=10, cv=10, scoring="roc_auc", max_iter=max_iter)
    return Pipeline(steps=[("sc", StandardScaler()), ("clf", lr_clf)])

# 4. evaluate classifier
def evaluate_link_prediction_model(
    clf, link_examples_test, link_labels_test, get_embedding, binary_operator
):
    link_features_test = link_examples_to_features(
        link_examples_test, get_embedding, binary_operator
    )
    score = evaluate_roc_auc(clf, link_features_test, link_labels_test)
    return score

def evaluate_roc_auc(clf, link_features, link_labels):
    predicted = clf.predict_proba(link_features)

    # check which class corresponds to positive links
    positive_column = list(clf.classes_).index(1)
    return roc_auc_score(link_labels, predicted[:, positive_column])


def main():
    parser = ArgumentParser("link prediction",
                        formatter_class=ArgumentDefaultsHelpFormatter,
                        conflict_handler='resolve')

    parser.add_argument('--operator',
                        help='Select binary operator from hadamard, l1, l2, average.')

    parser.add_argument("--method", required=True, default='n2v',
                        help='Embedding method.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter for node2vec. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter for node2vec. Default is 1.')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel processes. Default is 8.')

    parser.add_argument('--representation-size', type=int, default=128,
                        help='Number of dimensions to learn for each node. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument("--all", default=False, action='store_true',
                        help='The embeddings are evaluated on all training percents from 10 to 90 when this flag is set to true. '
                             'By default, only training percents of 10, 50 and 90 are used.')

    parser.add_argument("--network", required=True,
                        help='Network graph')

    args = parser.parse_args()

    # data
    matfile = args.network
    mat_varables = loadmat(matfile)
    x = mat_varables['network']

    G = Graph()
    if issparse(x):
        cx = x.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            G.add_edge(i, j, weight=1)
    else:
        raise Exception("Dense matrices not yet supported.")

    graph = StellarGraph.from_networkx(G)

    # graph_test: for computing test node embeddings with more edges than the graph_train
    # examples_test: test set of + and - edges not used for computing the test node embeddings or for classifier training or model selection
    edge_splitter_test = EdgeSplitter(graph)
    graph_test, examples_test, labels_test = edge_splitter_test.train_test_split(
        p=0.1, method="global"
    )

    # graph_train: for computing node embeddings
    # examples: training set of + and - edges not used for computing node embeddings
    edge_splitter_train = EdgeSplitter(graph_test, graph)
    graph_train, examples, labels = edge_splitter_train.train_test_split(
        p=0.1, method="global"
    )

    if args.method == 'n2v':
        embedding_train = node2vec_embedding(args, graph_train)
        embedding_test = node2vec_embedding(args, graph_test)

    #TODO: add other methods here

    # 1. calculate link/edge emb for + and - samples by applying a binary operator
    # 2. given the emb, train a logistic regression classifier to predict whether an edge between two nodes should exist or not.
    # 3. evaluate the performance of the classifier for each of the 4 operators on the training data with emb on graph_train, and select the best classifier
    # 4. the best classifier is then used to calculate scores on the test data with node embeddings calculated on graph_test

    if args.all:
        training_percents = np.asarray(range(1, 10)) * .1
    else:
        training_percents = [0.1, 0.5, 0.9]

    all_results = {}
    for train_percent in training_percents:
        (examples_train, examples_model_selection, labels_train, labels_model_selection,) = train_test_split(examples, labels, train_size=train_percent, test_size=1-train_percent)
        if args.operator == "l1":
            clf = train_link_prediction_model(
                examples_train, labels_train, embedding_train, operator_l1
            )
            score = evaluate_link_prediction_model(
                clf,
                examples_model_selection,
                labels_model_selection,
                embedding_train,
                operator_l1,
            )
        elif args.operator == "l2":
            clf = train_link_prediction_model(
                examples_train, labels_train, embedding_train, operator_l2
            )
            score = evaluate_link_prediction_model(
                clf,
                examples_model_selection,
                labels_model_selection,
                embedding_train,
                operator_l2,
            )
        elif args.operator =="hadamard":
            clf = train_link_prediction_model(
                examples_train, labels_train, embedding_train, operator_hadamard
            )
            score = evaluate_link_prediction_model(
                clf,
                examples_model_selection,
                labels_model_selection,
                embedding_train,
                operator_hadamard,
            )
        elif args.operator == "average":
            clf = train_link_prediction_model(
                examples_train, labels_train, embedding_train, operator_avg
            )
            score = evaluate_link_prediction_model(
                clf,
                examples_model_selection,
                labels_model_selection,
                embedding_train,
                operator_avg,
            )
        else:
            raise Exception("Unknown operator, select from l1, l2, hadamard, and average" % args.format)

        all_results[train_percent] = score

    eva_fname = 'evaluation/'+ matfile[matfile.find('/'):matfile.find('.mat')] + '-' + 'n2v.txt'
    sys.stdout = open(eva_fname, "w")

    print('Results, using embeddings of dimensionality', args.representation_size)
    print('Operator', args.operator)
    print('-------------------')
    for train_percent in sorted(all_results.keys()):
        print('Train percent:', train_percent)
        print(all_results[train_percent])
        print('-------------------')
    sys.stdout.close()

if __name__ == "__main__":
    sys.exit(main())
