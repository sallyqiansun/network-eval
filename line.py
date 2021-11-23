import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import networkx as nx
import pickle

class AliasSampling:

    # Reference: https://en.wikipedia.org/wiki/Alias_method

    def __init__(self, prob):
        self.n = len(prob)
        self.U = np.array(prob) * self.n
        self.K = [i for i in range(len(prob))]
        overfull, underfull = [], []
        for i, U_i in enumerate(self.U):
            if U_i > 1:
                overfull.append(i)
            elif U_i < 1:
                underfull.append(i)
        while len(overfull) and len(underfull):
            i, j = overfull.pop(), underfull.pop()
            self.K[j] = i
            self.U[i] = self.U[i] - (1 - self.U[j])
            if self.U[i] > 1:
                overfull.append(i)
            elif self.U[i] < 1:
                underfull.append(i)

    def sampling(self, n=1):
        x = np.random.rand(n)
        i = np.floor(self.n * x)
        y = self.n * x - i
        i = i.astype(np.int32)
        res = [i[k] if y[k] < self.U[i[k]] else self.K[i[k]] for k in range(n)]
        if n == 1:
            return res[0]
        else:
            return res


class Model:
    def __init__(self, config, G):
        self.u_i = tf.placeholder(name='u_i', dtype=tf.int32, shape=[config["batch-size"] * (config["K"] + 1)])
        self.u_j = tf.placeholder(name='u_j', dtype=tf.int32, shape=[config["batch-size"] * (config["K"] + 1)])
        self.label = tf.placeholder(name='label', dtype=tf.float32, shape=[config["batch-size"] * (config["K"] + 1)])
        self.embedding = tf.get_variable('target_embedding', [G.number_of_nodes(), config["dimensions"]],
                                         initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
        self.u_i_embedding = tf.matmul(tf.one_hot(self.u_i, depth=G.number_of_nodes()), self.embedding)
        if config["proximity"] == 'first-order':
            self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=G.number_of_nodes()), self.embedding)
        elif config["proximity"] == 'second-order':
            self.context_embedding = tf.get_variable('context_embedding', [G.number_of_nodes(), config["dimensions"]],
                                                     initializer=tf.random_uniform_initializer(minval=-1., maxval=1.))
            self.u_j_embedding = tf.matmul(tf.one_hot(self.u_j, depth=G.number_of_nodes()), self.context_embedding)

        self.inner_product = tf.reduce_sum(self.u_i_embedding * self.u_j_embedding, axis=1)
        self.loss = -tf.reduce_mean(tf.log_sigmoid(self.label * self.inner_product))
        self.learning_rate = tf.placeholder(name='learning_rate', dtype=tf.float32)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=config["learning-rate"])
        self.train_op = self.optimizer.minimize(self.loss)


def run(config, G):
    suffix = config["proximity"]
    edges = list(G.edges())
    nodes = list(G.nodes())

    node_index = {}
    node_index_reversed = {}
    for index, (node, _) in enumerate(edges):
        node_index[node] = index
        node_index_reversed[index] = node

    model = Model(config, G)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        initial_embedding = sess.run(model.embedding)
        learning_rate = config["learning-rate"]

        edge_distribution = np.array([1 for edge in edges], dtype=np.float32)
        edge_distribution /= np.sum(edge_distribution)
        edge_sampling = AliasSampling(prob=edge_distribution)
        node_negative_distribution = np.power(
            np.array([G.degree(node) for node in nodes], dtype=np.float32), 0.75)
        node_negative_distribution /= np.sum(node_negative_distribution)
        node_sampling = AliasSampling(prob=node_negative_distribution)

        for b in range(config["batch"]):
            edge_batch_index = edge_sampling.sampling(config["batch-size"])

            u_i = []
            u_j = []
            label = []
            for edge_index in edge_batch_index:
                edge = edges[edge_index]
                if G.__class__ == nx.Graph:
                    if np.random.rand() > 0.5:  # important: second-order proximity is for directed edge
                        edge = (edge[1], edge[0])
                u_i.append(edge[0])
                u_j.append(edge[1])
                label.append(1)
                for i in range(config["K"]):
                    while True:
                        negative_node = node_sampling.sampling()
                        if not G.has_edge(node_index_reversed[negative_node], node_index_reversed[edge[0]]):
                            break
                    u_i.append(edge[0])
                    u_j.append(negative_node)
                    label.append(-1)

            feed_dict = {model.u_i: u_i, model.u_j: u_j, model.label: label, model.learning_rate: learning_rate}

            if b % 100 != 0:
                sess.run(model.train_op, feed_dict=feed_dict)
                if learning_rate > learning_rate * 0.0001:
                    learning_rate = learning_rate * (1 - b / config["batch"])
                else:
                    learning_rate = learning_rate * 0.0001
            else:
                loss = sess.run(model.loss, feed_dict=feed_dict)
            if b % 1000 == 0 or b == (config["batch"] - 1):
                embedding = sess.run(model.embedding)
                normalized_embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
                mapped = {node: normalized_embedding[node-1] for node in nodes}
                f = open(config['emb-path'], "w")
                f.write("{} {}\n".format(len(nodes), len(mapped[nodes[0]])))
                for k in mapped:
                    f.write("{} ".format(k))
                    for emb in mapped[k]:
                        f.write("{} ".format(emb))
                    f.write("\n")
                f.close()