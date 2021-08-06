import networkx as nx
import argparse
from run import read_graph
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser(description="Convert .mat file to adjlist and edgelist graphs.")

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


if __name__ == "__main__":
    args = parse_args()
    args.format = 'mat'
    main(args)
