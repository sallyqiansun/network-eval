import numpy as np
import random
import run

class Graph():
	def __init__(self, alias_nodes, alias_edges):
		super(Graph, self).__init__(list)
		# self.alias_nodes = alias_nodes
		# self.alias_edges = alias_edges


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

def get_alias_edge(G, src, dst, args):
	'''
	Get the alias edge setup lists for a given edge.
	'''
	p = args.p
	q = args.q

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
	triads = {}

	if args.directed:
		for edge in G.edges():
			alias_edges[edge] = get_alias_edge(G, edge[0], edge[1], args)
	else:
		for edge in G.edges():
			alias_edges[edge] = get_alias_edge(G, edge[0], edge[1], args)
			alias_edges[(edge[1], edge[0])] = get_alias_edge(G, edge[1], edge[0], args)

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





def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	G = run.read_graph(args)
	run.simulate_and_embed(args, G=G)


if __name__ == "__main__":
	args = run.parse_args()
	main(args)
