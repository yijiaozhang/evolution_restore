import numpy as np
import random


class Graph():
    def __init__(self, nx_g, is_directed, p, q):
        self.g = nx_g
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        """
        simulate a random walk starting from start node.
        """
        g = self.g
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(g.neighbors(cur))
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

    def simulate_walks(self, num_walks, walk_length):
        """
        repeatedly simulate random walks from each node.
        """
        g = self.g
        walks = []
        nodes = list(g.nodes())
        print('walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        """
        get the alias edge setup lists for a given edge.
        """
        g = self.g
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(g.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(g[dst][dst_nbr]['weight']/p)
            elif g.has_edge(dst_nbr, src):
                unnormalized_probs.append(g[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(g[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
        preprocessing of transition probabilities for guiding the random walks.
        """
        g = self.g
        is_directed = self.is_directed

        alias_nodes = {}
        for node in g.nodes():
            unnormalized_probs = [g[node][nbr]['weight'] for nbr in sorted(g.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in g.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in g.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    """
    compute utility lists for non-uniform sampling from discrete distributions.
    refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    k = len(probs)
    q = np.zeros(k)
    j = np.zeros(k, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = k*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        j[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return j, q


def alias_draw(j, q):
    """
    draw sample from a non-uniform discrete distribution using alias sampling.
    """
    k = len(j)

    kk = int(np.floor(np.random.rand()*k))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return j[kk]
