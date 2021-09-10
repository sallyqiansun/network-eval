import argparse
import networkx as nx
from gensim.models import Word2Vec
from networkx import Graph
from scipy.io import loadmat
from scipy.sparse import issparse
import random
import numpy as np


def parse_args():
	parser = argparse.ArgumentParser(description="Run embedding methods.")

	parser.add_argument('--input', nargs='?', default='examples/karate.edgelist',
						help='Input graph path')

	parser.add_argument('--output', nargs='?', default='embedding/karate.emb',
						help='Embeddings path')

	parser.add_argument('--method', nargs='?', default='node2vec',
						help='Network embedding method')

	parser.add_argument('--format', nargs='?', default='edgelist',
						help='File format of input file')

	parser.add_argument('--dimensions', type=int, default=128,
						help='Number of dimensions to learn for each node. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
						help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
						help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
						help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
						help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
						help='Number of parallel processes. Default is 8.')

	parser.add_argument('--seed', default=0, type=int,
						help='Seed for random walk generator.')

	parser.add_argument('--p', type=float, default=1,
						help='Return hyperparameter for node2vec. Default is 1, the case for deepwalk.')

	parser.add_argument('--q', type=float, default=1,
						help='Inout hyperparameter for node2vec. Default is 1, the case for deepwalk.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
						help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
						help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

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

def preprocess_transition_probs(G, args):
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

	if args.directed:
		for edge in G.edges():
			alias_edges[edge] = get_alias_edge(G, edge[0], edge[1], p=args.p, q=args.q)
	else:
		for edge in G.edges():
			alias_edges[edge] = get_alias_edge(G, edge[0], edge[1], p=args.p, q=args.q)
			alias_edges[(edge[1], edge[0])] = get_alias_edge(G, edge[1], edge[0], p=args.p, q=args.q)

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


def read_graph(format):
	'''
	Reads the input network in networkx.
	'''
	if format == "edgelist":
		G = read_graph_edgelist(args)
	elif format == "adjlist":
		G = read_graph_adjlist(args)
	elif format == "mat":
		G = read_graph_mat(args)
	else:
		raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)

	return G

def read_graph_edgelist(args):
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),))
	else:
		G = nx.read_edgelist(args.input, nodetype=int)
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def read_graph_adjlist(args):
	G = nx.read_adjlist(args.input, nodetype=int)

	if not args.directed:
		G = G.to_undirected()

	return G

def read_graph_mat(args, variable_name='network'):
  mat_varables = loadmat(args.input)
  x = mat_varables[variable_name]

  G = Graph()

  if issparse(x):
	  cx = x.tocoo()
	  for i, j, v in zip(cx.row, cx.col, cx.data):
		  G.add_edge(i, j, weight=1)
  else:
	  raise Exception("Dense matrices not yet supported.")

  if not args.directed:
	  G.to_undirected()

  return G

def simulate_and_embed(args, G):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''

	rand = random.Random(args.seed)
	print("Number of nodes: {}".format(len(G.nodes())))

	num_walks = len(G.nodes()) * args.num_walks
	print("Number of walks: {}".format(num_walks))

	data_size = num_walks * args.walk_length
	print("Data size (walks*length): {}".format(data_size))

	if args.method == "node2vec":
		preprocess_transition_probs(G, args)

		print("Walking...")
		walks = simulate_walks(G, num_walks=args.num_walks, walk_length=args.walk_length)
		walks = [list(map(str, walk)) for walk in walks]

		print("Training...")
		model = Word2Vec(walks, window=args.window_size, min_count=0, sg=1, workers=args.workers, epochs=args.iter)
		model.iter = args.iter

	elif args.method == "deepwalk":
		if args.p != 1 or args.q != 1:
			raise Exception("For deepwalk, p=q=1.")
		else:
			preprocess_transition_probs(G, args)

			print("Walking...")
			walks = simulate_walks(G, num_walks=args.num_walks, walk_length=args.walk_length)
			walks = [list(map(str, walk)) for walk in walks]

			print("Training...")
			model = Word2Vec(walks, window=args.window_size, min_count=0, sg=1, workers=args.workers, epochs=args.iter)
			model.iter = args.iter

	else:
		raise Exception("This embedding method only supports deepwalk and node2vec.")

	model.wv.save_word2vec_format(args.output)
	print("Embedding saved to {}.".format(args.output))
	return

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	G = read_graph(args.format)
	simulate_and_embed(args, G=G)


if __name__ == "__main__":
	args = parse_args()
	main(args)
