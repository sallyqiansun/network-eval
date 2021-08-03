import argparse
import networkx as nx
from gensim.models import Word2Vec
from deepwalk import *

def parse_args():
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run embedding methods.")

	parser.add_argument('--input', nargs='?', default='examples/karate.edgelist',
						help='Input graph path')

	parser.add_argument('--output', nargs='?', default='output/karate-dw.emb',
						help='Embeddings path')

	parser.add_argument('--method', nargs='?', default='node2vec',
						help='Network embedding method')

	parser.add_argument('--format', nargs='?', default='edgelist',
						help='File format of input file')

	parser.add_argument('--dimensions', type=int, default=128,
						help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
						help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
						help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
						help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
						help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
						help='Number of parallel workers. Default is 8.')

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
	if args.method == "node2vec":
		read_graph_n2v(args)
	elif args.method == "deepwalk":
		read_graph_dw(args)
	else:
		raise Exception("Will add other methods here")


def read_graph_n2v(args):
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

#TODO: complete this
def read_graph_dw(args)

def learn_embeddings(args, walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [list(map(str, walk)) for walk in walks]
	model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers)
	model.iter = args.iter
	model.wv.save_word2vec_format(args.output)

	print ('Embedding saved to: ', args.output)
	return