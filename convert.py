import networkx as nx
import argparse
from run import read_graph
from scipy import io

def parse_args():

    parser = argparse.ArgumentParser(description="Convert adjlist or edgelist graph to sparse matrix .mat file")

    parser.add_argument('--input', nargs='?', default='examples/karate.edgelist', help='Input graph path')

    parser.add_argument('--format', nargs='?', default='edgelist', help='Select from edgelist and adjlist')

    return parser.parse_args()

def main(args):
    in_fname = args.input
    G = read_graph(args)
    M = nx.to_scipy_sparse_matrix(G, dtype=int, format='coo')
    io.savemat('output/'+in_fname[in_fname.find('/'):in_fname.find('.')]+'-converted.mat', {'M': M})
