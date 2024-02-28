"""
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
"""

import argparse
import numpy as np
import networkx as nx
from ._node2vec import Graph
from gensim.models import Word2Vec


def parse_args():
    """
    Parses the node2vec arguments.
    """
    parser = argparse.ArgumentParser(description="Run node2vec.")

    parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,  # walk_length游走深度，初始每个节点每次游走的深度80
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,  # num_walks游走次数，初始每个节点进行10次游走  随机游走的时候每个顶点被选取作为初始点的次数
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph():
    """
    Reads the input network in network.
    """
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings(walks):
    """
    Learn embeddings by optimizing the Skipgram objective using SGD.
    """
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers,
                     iter=args.iter)
    model.wv.save_word2vec_format(args.output)
    return


def main(args):
    """
    Pipeline for representational learning for all nodes in a graph.
    """
    nx_G = read_graph()
    G = Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks)


def my_main(edges: object, output: object, directed: object = False, p: object = 1, q: object = 1,
            num_walks: object = 10, walk_length: object = 80, dimensions: object = 128,
            window_size: object = 10,
            workers: object = 8,
            _iter: object = 1) -> object:
    """

    :param edges:
    :param output:
    :param directed:
    :param p:
    :param q:
    :param num_walks:
    :param walk_length:
    :param dimensions:
    :param window_size:
    :param workers:
    :param _iter:
    :rtype: object
    """
    nx_G = nx.DiGraph()
    nx_G.add_edges_from(edges)
    for edge in nx_G.edges():
        nx_G[edge[0]][edge[1]]['weight'] = 1
    nx_G = nx_G.to_undirected()

    G = Graph(nx_G, directed, p, q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_length)
    # learn_embeddings(walks)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, workers=workers, iter=_iter)
    model.wv.save_word2vec_format(output)


if __name__ == "__main__":
    args = parse_args()
    main(args)
