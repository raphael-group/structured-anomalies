import numpy as np, networkx as nx
import sys, argparse

from common import load_node_score, load_edge_list, save_nodes, load_matrix

# Parse arguments.
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True, help='Input score file (.tsv)')
    parser.add_argument('-elf', '--edge_list_file', type=str, required=False, help='Input edge list file (.tsv)')
    parser.add_argument('-o', '--output_file', type=str, required=True, help='Output anomaly file (.tsv)')
    parser.add_argument('-a', '--anomaly_type', type=str, choices=['cutsize', 'line', 'connected', 'unconstrained', 'submatrix'], default='connected', help='Type of anomaly')
    parser.add_argument('-m', '--matrix_name', type=str, required=False, help='Name of matrix')
    return parser

def run(args):

    if args.edge_list_file is None:
        if args.anomaly_type == 'cutsize' or args.anomaly_type == 'line' or args.anomaly_type == 'connected':
            raise ValueError('Anomaly of type {} needs edge list file'.format(args.anomaly_type))
    else: 
        edge_list = load_edge_list(args.edge_list_file)
        G = nx.Graph()
        G.add_edges_from(edge_list)
        G = G.subgraph(set_nodes)

    if args.anomaly_type == 'submatrix':
        if args.matrix_name is None:
            A = load_matrix(args.input_file)
        else:
            A = load_matrix(args.input_file, args.matrix_name)

        

    elif args.anomaly_type == 'connected':
        # NetMix code here
    elif args.anomaly_type == 'line':
        names, scores = load_node_score(args.input_file)

    elif args.anomaly_type == 'unconstrained':
        names, scores = load_node_score(args.input_file)

    elif args.anomaly_type = 'cutsize':
        

if __name__=='__main__':
    run(get_parser().parse_args(sys.argv[1:]))