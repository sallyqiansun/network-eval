import argparse
import networkx as nx
from scipy.io import loadmat
from scipy.sparse import issparse
from deepwalk import *
from node2vec import *
from networkx import Graph
def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run embedding methods.")

	parser.add_argument('--input', nargs='?', default='examples/karate.edgelist',
						help='Input graph path')

	parser.add_argument('--output', nargs='?', default='output/karate-n2v.emb',
						help='Embeddings path')

	parser.add_argument('--method', nargs='?', default='node2vec',
						help='Network embedding method')

	parser.add_argument('--format', nargs='?', default='edgelist',
						help='File format of input file')

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

	parser.add_argument('--workers', type=int, default=8,
						help='Number of parallel processes. Default is 8.')

	parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
						help='Size to start dumping walks to disk, instead of keeping them in memory.')

	parser.add_argument('--seed', default=0, type=int,
						help='Seed for random walk generator.')

	parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
						help='Use vertex degree to estimate the frequency of nodes '
							 'in the random walks. This option is faster than '
							 'calculating the vocabulary.')

	parser.add_argument('--p', type=float, default=1,
						help='Return hyperparameter for node2vec. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
						help='Inout hyperparameter for node2vec. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
						help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
						help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()


def read_graph(args):
	'''
	Reads the input network in networkx.
	'''
	if args.format == "edgelist":
		G = read_graph_edgelist(args)
	elif args.format == "adjlist":
		G = read_graph_adjlist(args)
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

	if args.method == "deepwalk":
		if data_size < args.max_memory_data_size:
			print("Walking...")
			walks = build_deepwalk_corpus(G, num_paths=args.num_walks,
										  path_length=args.walk_length, alpha=0, rand=rand)
			print("Training...")
			model = Word2Vec(walks, vector_size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1,
							 workers=args.workers)
		else:
			print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(data_size,
																												 args.max_memory_data_size))
			print("Walking...")

			walks_filebase = args.output + ".walks"
			walk_files = write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
															  path_length=args.walk_length, alpha=0,
															  rand=rand,
															  num_workers=args.workers)

			print("Counting vertex frequency...")
			if not args.vertex_freq_degree:
				vertex_counts = count_textfiles(walk_files, args.workers)
			else:
				# use degree distribution for frequency in tree
				vertex_counts = G.degree(nodes=G.iterkeys())

			print("Training...")
			walks_corpus = WalksCorpus(walk_files)
			model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
							 size=args.representation_size,
							 window=args.window_size, min_count=0, trim_rule=None, workers=args.workers)

	elif args.method == "node2vec":
		preprocess_transition_probs(G, args)

		print("Walking...")
		walks = simulate_walks(G, num_walks=args.num_walks, walk_length=args.walk_length)
		walks = [list(map(str, walk)) for walk in walks]

		print("Training...")
		model = Word2Vec(walks, window=args.window_size, min_count=0, sg=1, workers=args.workers)
		model.iter = args.iter

	model.wv.save_word2vec_format(args.output)
	print("Embedding saved to {}.".format(args.output))
	return