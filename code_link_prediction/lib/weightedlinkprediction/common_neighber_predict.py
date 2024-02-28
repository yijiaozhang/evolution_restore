import numpy as np
import networkx as nx
from math import log
import weightedlinkprediction.utils
logger = weightedlinkprediction.utils.get_logger(__name__)

class CommonNeighber(object):
    """
    Common neighber Link prediction object
    Given a network, compute the common neighber index of all node pairs in ebunch (all non-edges).
    Reference: 
        Ahmad, I., Akhtar, M.U., Noor, S. et al.
        Missing Link Prediction using Common Neighbor and Centrality based Parameterized Algorithm.
        Sci Rep 10, 364 (2020). https://doi.org/10.1038/s41598-019-57304-y
    """

    def __init__(
        self,
        G,
        func,
        removed_edges,
    ):
        """
        :param G: network with weighted edges
        :param func: methods of link prediction: "predict" or "predict_weighted"
        :param removed_edges: list of removed edges
        """
        self.G = G
        self.func = func
        self.removed_edges = removed_edges
        self.predicted_edges = []
        self.hit_result = []

    def hit(self):
        self.predicted_edges = self.common_neighber()
        for e in self.predicted_edges:
            if (e[0], e[1]) in self.removed_edges or (e[1], e[0]) in self.removed_edges:
                self.hit_result.append(1)
            else:
                self.hit_result.append(0)
        self.hit_result = np.array(self.hit_result)
        return self.hit_result

    def predict(self, u, v):
        return sum(1 for w in nx.common_neighbors(self.G, u, v))

    def predict_weighted(self, u, v):
        return sum(self.G.edges[w, u]['weight'] + self.G.edges[w, v]['weight'] for w in nx.common_neighbors(self.G, u, v))

    def common_neighber(self):
        ebunch = nx.non_edges(self.G)
        if self.func == 'predict':
            temp_cn_predict = ((u, v, self.predict(u, v)) for u, v in ebunch)  
        elif self.func == 'predict_weighted':
            temp_cn_predict = ((u, v, self.predict_weighted(u, v)) for u, v in ebunch)  
        else:
            temp_cn_predict = ()
            logger.info('wrong func')
        return sorted(list(temp_cn_predict), key=lambda tup: tup[2], reverse=True)[:len(self.removed_edges)]

