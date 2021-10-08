import networkx as nx
import numpy as np
from scipy.io import loadmat
from scipy.sparse import issparse
from networkx import Graph

def parse_args():

    parser = argparse.ArgumentParser(description="Supports the following conversion of file formats： .mat->.adjlist & .edgelist; .csv->.edgelist")

    parser.add_argument('--input', nargs='?', default='examples/blogcatalog.mat', help='Input .mat path')

    parser.add_argument('--weighted', dest='weighted', action='store_true', help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true', help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def main(args):
    in_fname = args.input
    G = read_graph(args)

    edgelist_fname = 'converted/'+in_fname[in_fname.find('/')+1:in_fname.find('.')]+'-converted.edgelist'
    adjlist_fname = 'converted/' + in_fname[in_fname.find('/') + 1:in_fname.find('.')] + '-converted.adjlist'

    e_list = nx.convert.to_edgelist(G)
    a_list = nx.to_dict_of_lists(G)

    edgelist_out = open(edgelist_fname, "w")
    for edge in e_list:
        edgelist_out.write(' '.join(str(e) for e in edge) + '\n')
    edgelist_out.close()
    print("file saved to {}".format(edgelist_fname))

    adjlist_out = open(adjlist_fname, "w")
    for adj in a_list.keys():
        adjlist_out.write(str(adj) + ' ' + ' '.join(str(a) for a in a_list[adj]) + '\n')
    adjlist_out.close()
    print("file saved to {}".format(adjlist_fname))


def read_graph(format, input, directed, weighted, variable_name):
    '''
    Reads the input network in networkx.
    '''
    if format == "edgelist":
        G = read_graph_edgelist(input, directed, weighted)
    elif format == "adjlist":
        G = read_graph_adjlist(input, directed, weighted)
    elif format == "mat":
        G = read_graph_mat(input, directed, weighted, variable_name)
    else:
        raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % format)

    return G


def read_graph_edgelist(input, directed, weighted):
    if weighted:
        G = nx.read_edgelist(input, nodetype=int, data=(('weight', float),))
    else:
        G = nx.read_edgelist(input, nodetype=int)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if directed == False:
        G = G.to_undirected()

    return G


def read_graph_adjlist(input, directed, weighted):
    G = nx.read_adjlist(input, nodetype=int)

    if directed == False:
        G = G.to_undirected()

    return G


def read_graph_mat(input, directed, weighted, variable_name):
    mat_varables = loadmat(input)
    x = mat_varables[variable_name]

    G = Graph()

    if issparse(x):
        cx = x.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            G.add_edge(i, j, weight=1)
    else:
        raise Exception("Dense matrices not yet supported.")

    if directed == False:
        G.to_undirected()

    return G
