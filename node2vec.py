import argparse
import networkx as nx
from gensim.models import Word2Vec
from networkx import Graph
from scipy.io import loadmat
from scipy.sparse import issparse
import random
import numpy as np


def node2vec_walk(G, walk_length, start_node):
	'''
	Simulate a random walk starting from start node.
	'''
	alias_nodes = G.alias_nodes
	alias_edges = G.alias_edges

	walk = [start_node]

	while len(walk) < walk_length:
		cur = walk[-1]
		cur_nbrs = sorted(G.neighbors(cur))
		if len(cur_nbrs) > 0:
			if len(walk) == 1:
				walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
			else:
				prev = walk[-2]
				next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],
					alias_edges[(prev, cur)][1])]
				walk.append(next)
		else:
			break

	return walk

def simulate_walks(G, num_walks, walk_length):
	'''
	Repeatedly simulate random walks from each node.
	'''
	walks = []
	nodes = list(G.nodes())
	print ('Walk iteration:')
	for walk_iter in range(num_walks):
		print (str(walk_iter+1), '/', str(num_walks))
		random.shuffle(nodes)
		for node in nodes:
			walks.append(node2vec_walk(G, walk_length=walk_length, start_node=node))

	return walks

def get_alias_edge(G, src, dst, p, q):
	'''
	Get the alias edge setup lists for a given edge.
	'''

	unnormalized_probs = []
	for dst_nbr in sorted(G.neighbors(dst)):
		if dst_nbr == src:
			unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
		elif G.has_edge(dst_nbr, src):
			unnormalized_probs.append(G[dst][dst_nbr]['weight'])
		else:
			unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
	norm_const = sum(unnormalized_probs)
	normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

	return alias_setup(normalized_probs)

def preprocess_transition_probs(G, config):
	'''
	Preprocessing of transition probabilities for guiding the random walks.
	'''

	alias_nodes = {}
	for node in G.nodes():
		unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
		alias_nodes[node] = alias_setup(normalized_probs)

	alias_edges = {}

	if config['directed']:
		for edge in G.edges():
			alias_edges[edge] = get_alias_edge(G, edge[0], edge[1], p=config['p'], q=config['q'])
	else:
		for edge in G.edges():
			alias_edges[edge] = get_alias_edge(G, edge[0], edge[1], p=config['p'], q=config['q'])
			alias_edges[(edge[1], edge[0])] = get_alias_edge(G, edge[1], edge[0], p=config['p'], q=config['q'])

	G.alias_nodes = alias_nodes
	G.alias_edges = alias_edges

	return

def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K*prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()

		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)

	return J, q

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
		return kk
	else:
		return J[kk]



def simulate_and_embed(config, G, workers=8):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''

	rand = random.Random(config['seed'])
	print("Number of nodes: {}".format(len(G.nodes())))

	num_walks = len(G.nodes()) * config['num-walks']
	print("Number of walks: {}".format(num_walks))

	data_size = num_walks * config['walk-size']
	print("Data size (walks*length): {}".format(data_size))

	if config['method'] == "node2vec":
		preprocess_transition_probs(G, config)

		print("Walking...")
		walks = simulate_walks(G, num_walks=config['num-walks'], walk_length=config['walk-size'])
		walks = [list(map(str, walk)) for walk in walks]

		print("Training...")
		model = Word2Vec(walks, window=config['window-size'], min_count=0, sg=1, workers=workers, epochs=config['iter'])

	elif config['method'] == "deepwalk":
		if config['p'] != 1 or config['q'] != 1:
			raise Exception("For deepwalk, p=q=1.")
		else:
			preprocess_transition_probs(G, config)

			print("Walking...")
			walks = simulate_walks(G, num_walks=config['num-walks'], walk_length=config['walk-size'])
			walks = [list(map(str, walk)) for walk in walks]

			print("Training...")
			model = Word2Vec(walks, window=config['window-size'], min_count=0, sg=1, workers=workers, epochs=config['iter'])

	else:
		raise Exception("This embedding method only supports deepwalk and node2vec.")

	model.wv.save_word2vec_format(config['emb-path'])
	print("Embedding saved to {}.".format(config['emb-path']))
	return


