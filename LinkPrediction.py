import networkx as nx
import numpy as np
import random as rand
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics, model_selection, pipeline
from collections import defaultdict
from sklearn.metrics import roc_auc_score

# codes for train-test split and pos-neg edges generation are adapted from https://github.com/adocherty/node2vec_linkprediction
def generate_pos_neg_links(G):
    """
    Select random existing edges in the graph to be postive links,
    and random non-edges to be negative links. proportion is 1:1
    Modify graph by removing the postive links.
    This function returns the selected links, and their corresponding labels
    """
    # Select n edges at random (positive samples)
    n_edges = len(G.edges())
    n_nodes = len(G.nodes())
    npos = int(0.5 * n_edges)
    nneg = n_edges - npos

    n_neighbors = [len(list(G.neighbors(v))) for v in G.nodes()]
    n_non_edges = n_nodes - np.array(n_neighbors) - 1

    non_edges = [e for e in nx.non_edges(G)]
    print("Finding %d of %d non-edges" % (nneg, len(non_edges)))

    # Select m pairs of non-edges (negative samples)
    rnd_inx = rand.sample(range(0, len(non_edges)), nneg)
    neg_edge_list = [non_edges[i] for i in rnd_inx]

    if len(neg_edge_list) < nneg:
        raise RuntimeWarning(
            "Only %d negative edges found" % (len(neg_edge_list))
        )
    else:
        print("Found %d negative edges of %d total edges" % (len(neg_edge_list), n_edges))

    # Find positive edges, and remove them.
    edges = list(G.edges())
    pos_edge_list = []
    n_count = 0
    n_ignored_count = 0
    rnd_inx = np.random.permutation(n_edges)
    for i in rnd_inx:
        edge = edges[i]
        n1 = edge[0]
        n2 = edge[1]

        # Remove edge from graph
        G.remove_edge(n1, n2)

        # Check if graph is still connected
        reachable_from_v1 = nx.connected._plain_bfs(G, n1)
        if n1 not in reachable_from_v1:
            G.add_edge(n1, n2)
            n_ignored_count += 1
        else:
            pos_edge_list.append([n1, n2])
            n_count += 1

        # Exit if we've found npos nodes or we have gone through the whole list
        if n_count >= npos:
            break

    if len(pos_edge_list) < npos:
        raise RuntimeWarning("Only %d positive edges found." % (n_count))

    return pos_edge_list, neg_edge_list


def run(graph, config, method="hadamard"):
    all_results = defaultdict(list)
    # 1. get embeddings graph ready
    embeddings_file = config['emb-path']
    emb = KeyedVectors.load(embeddings_file, mmap='r')

    # # 2. Set positive and negative training sets
    for train_percent in config['train_percent']:
        pos_links, neg_links = generate_pos_neg_links(graph)
        # train-test for pos_links
        train_num = int(train_percent*len(pos_links))
        rnd_inx = rand.sample(range(0, len(pos_links)), train_num)
        train_pos_links = [pos_links[i] for i in rnd_inx]
        train_pos_lables = [1] * len(train_pos_links)
        test_pos_links = [link for link in pos_links if link not in train_pos_links]
        test_pos_lables = [1] * len(test_pos_links)

        # train-test for neg_links
        rnd_inx = rand.sample(range(0, len(pos_links)), len(pos_links)-train_num)
        train_neg_links = [neg_links[i] for i in rnd_inx]
        train_neg_lables = [0] * len(train_neg_links)
        test_neg_links = list(set(neg_links) - set(train_neg_links))
        test_neg_lables = [0] * len(test_neg_links)

        train_links = train_pos_links + train_neg_links
        train_feat = get_features(train_links, emb, method)
        train_labels = train_pos_lables + train_neg_lables
        test_links = test_pos_links + test_neg_links
        test_labels = test_pos_lables + test_neg_lables
        test_feat = get_features(test_links, emb, method)

        scaler = StandardScaler()
        lin_clf = LogisticRegression(C=1)
        clf = pipeline.make_pipeline(scaler, lin_clf)

        clf.fit(train_feat, train_labels)
        auc_test = roc_auc_score(clf.predict(test_feat), test_labels)
        all_results[train_percent].append(auc_test)

    return all_results


def get_features(links, emb, method):
    features = []
    for link in links:
        emb1 = np.asarray(emb[str(link[0])])
        emb2 = np.asarray(emb[str(link[1])])

        # Calculate edge feature
        if method == "hadamard":
            features.append(emb1 * emb2)
        elif method == "average":
            features.append((emb1+emb2)*0.5)
        elif method == "l1":
            features.append(np.abs(emb1-emb2))
        elif method == "l2":
            features.append((emb1-emb2)**2)
        else:
            raise Exception("Method should be from hadamard, average, l1, and l2")

    return features
