import numpy as np
from scipy.sparse import csr_matrix
import random
import keras.backend as K
from sklearn import svm
from sklearn.metrics import accuracy_score
import math
import csv
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.layers import Embedding, Reshape, Merge, Activation, Input, merge
from tensorflow.keras.models import Sequential, Model



# code adapted from the python implementation of LINE, from https://github.com/VahidooX/LINE

def create_model(numNodes, factors):

    left_input = Input(shape=(1,))
    right_input = Input(shape=(1,))

    left_model = Sequential()
    left_model.add(Embedding(input_dim=numNodes + 1, output_dim=factors, input_length=1, mask_zero=False))
    left_model.add(Reshape((factors,)))

    right_model = Sequential()
    right_model.add(Embedding(input_dim=numNodes + 1, output_dim=factors, input_length=1, mask_zero=False))
    right_model.add(Reshape((factors,)))

    left_embed = left_model(left_input)
    right_embed = left_model(right_input)

    left_right_dot = merge([left_embed, right_embed], mode="dot", dot_axes=1, name="left_right_dot")

    model = Model(input=[left_input, right_input], output=[left_right_dot])
    embed_generator = Model(input=[left_input, right_input], output=[left_embed, right_embed])

    return model, embed_generator

def load_data(label_file, edge_file):
    csvfile = open(label_file, 'r')
    label_data = csv.reader(csvfile, delimiter=' ')
    labels_dict = dict()
    for row in label_data:
        labels_dict[int(row[0])] = int(row[1])

    csvfile = open(edge_file, 'r')
    adj_data = csv.reader(csvfile, delimiter=' ')
    adj_list = None
    for row in adj_data:
        for j in range(1, len(row)):
            if len(row[j]) == 0:
                continue
            a = int(row[0])
            b = int(row[j])

            if adj_list is None:
                adj_list = np.zeros((1, 2), dtype=np.int32)
                adj_list[0, :] = [a, b]
            else:
                adj_list = np.concatenate((adj_list, [[a, b]]), axis=0)

    adj_list = np.asarray(adj_list, dtype=np.int32)

    labeler = LabelEncoder()
    labeler.fit(list(set(adj_list.ravel())))

    adj_list = (labeler.transform(adj_list.ravel())).reshape(-1, 2)

    labels_dict = {labeler.transform([k])[0]: v for k, v in labels_dict.items() if k in labeler.classes_}

    return adj_list, labels_dict


def LINE_loss(y_true, y_pred):
    coeff = y_true*2 - 1
    return -K.mean(K.log(K.sigmoid(coeff*y_pred)))


def batchgen_train(adj_list, numNodes, batch_size, negativeRatio, negative_sampling):

    table_size = 1e8
    power = 0.75
    sampling_table = None

    data = np.ones((adj_list.shape[0]), dtype=np.int8)
    mat = csr_matrix((data, (adj_list[:,0], adj_list[:,1])), shape = (numNodes, numNodes), dtype=np.int8)
    batch_size_ones = np.ones((batch_size), dtype=np.int8)

    nb_train_sample = adj_list.shape[0]
    index_array = np.arange(nb_train_sample)

    nb_batch = int(np.ceil(nb_train_sample / float(batch_size)))
    batches = [(i * batch_size, min(nb_train_sample, (i + 1) * batch_size)) for i in range(0, nb_batch)]

    if negative_sampling == "NON-UNIFORM":
        print("Pre-procesing for non-uniform negative sampling!")
        node_degree = np.zeros(numNodes)

        for i in range(len(adj_list)):
            node_degree[adj_list[i,0]] += 1
            node_degree[adj_list[i,1]] += 1

        norm = sum([math.pow(node_degree[i], power) for i in range(numNodes)])

        sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        for j in range(numNodes):
            p += float(math.pow(node_degree[j], power)) / norm
            while i < table_size and float(i) / table_size < p:
                sampling_table[i] = j
                i += 1

    while 1:

        for batch_index, (batch_start, batch_end) in enumerate(batches):
            pos_edge_list = index_array[batch_start:batch_end]
            pos_left_nodes = adj_list[pos_edge_list, 0]
            pos_right_nodes = adj_list[pos_edge_list, 1]

            pos_relation_y = batch_size_ones[0:len(pos_edge_list)]

            neg_left_nodes = np.zeros(len(pos_edge_list)*negativeRatio, dtype=np.int32)
            neg_right_nodes = np.zeros(len(pos_edge_list)*negativeRatio, dtype=np.int32)

            neg_relation_y = np.zeros(len(pos_edge_list)*negativeRatio, dtype=np.int8)

            h = 0
            for i in pos_left_nodes:
                for k in range(negativeRatio):
                    rn = sampling_table[random.randint(0, table_size - 1)] if negative_sampling == "NON-UNIFORM" else random.randint(0, numNodes - 1)
                    while mat[i, rn] == 1 or i == rn:
                        rn = sampling_table[random.randint(0, table_size - 1)] if negative_sampling == "NON-UNIFORM" else random.randint(0, numNodes - 1)
                    neg_left_nodes[h] = i
                    neg_right_nodes[h] = rn
                    h += 1

            left_nodes = np.concatenate((pos_left_nodes, neg_left_nodes), axis=0)
            right_nodes = np.concatenate((pos_right_nodes, neg_right_nodes), axis=0)
            relation_y = np.concatenate((pos_relation_y, neg_relation_y), axis=0)

            yield ([left_nodes, right_nodes], [relation_y])


def svm_classify(X, label, split_ratios, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor for SVM
    """
    train_size = int(len(X)*split_ratios[0])
    val_size = int(len(X)*split_ratios[1])

    train_data, valid_data, test_data = X[0:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
    train_label, valid_label, test_label = label[0:train_size], label[train_size:train_size + val_size], label[train_size + val_size:]

    print('training SVM...')
    clf = svm.SVC(C=C, kernel='linear')
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(train_data)
    train_acc = accuracy_score(train_label, p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)
    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)

    return [train_acc, valid_acc, test_acc]


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret


if __name__ == "__main__":

    label_file = 'dblp/labels.txt'
    edge_file = 'dblp/adjedges.txt'
    epoch_num = 100
    factors = 128
    batch_size = 1000
    negative_sampling = "UNIFORM" # UNIFORM or NON-UNIFORM
    negativeRatio = 5
    split_ratios = [0.6, 0.2, 0.2]
    svm_C = 0.1

    np.random.seed(2017)
    random.seed(2017)

    adj_list, labels_dict = load_data(label_file, edge_file)
    epoch_train_size = (((int(len(adj_list)/batch_size))*(1 + negativeRatio)*batch_size) + (1 + negativeRatio)*(len(adj_list)%batch_size))
    numNodes = np.max(adj_list.ravel()) + 1
    data_gen = batchgen_train(adj_list, numNodes, batch_size, negativeRatio, negative_sampling)

    model, embed_generator = create_model(numNodes, factors)
    model.summary()

    model.compile(optimizer='rmsprop', loss={'left_right_dot': LINE_loss})

    model.fit_generator(data_gen, samples_per_epoch=epoch_train_size, nb_epoch=epoch_num, verbose=1)

    new_X = []
    new_label = []

    keys = list(labels_dict.keys())
    np.random.shuffle(keys)

    for k in keys:
        v = labels_dict[k]
        x = embed_generator.predict_on_batch([np.asarray([k]), np.asarray([k])])
        new_X.append(x[0][0] + x[1][0])
        new_label.append(labels_dict[k])

    new_X = np.asarray(new_X, dtype=np.float32)
    new_label = np.asarray(new_label, dtype=np.int32)

    [train_acc, valid_acc, test_acc] = svm_classify(new_X, new_label, split_ratios, svm_C)

    print("Train Acc:", train_acc, " Validation Acc:", valid_acc, " Test Acc:", test_acc)