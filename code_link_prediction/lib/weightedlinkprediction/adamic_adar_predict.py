import numpy as np
import networkx as nx
from math import log
import weightedlinkprediction.utils
logger = weightedlinkprediction.utils.get_logger(__name__)

class AdamicAdar(object):
    """
    Adamic Adar Link prediction object
    Given a network, compute the Adamic-Adar index of all node pairs in ebunch (all non-edges).
    Adamic Adar method in Reference: 
        D. Liben-Nowell, J. Kleinberg. The Link Prediction Problem for Social Networks (2004). 
        http://www.cs.cornell.edu/home/kleinber/link-pred.pdf
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
        self.predicted_edges = self.adamic_adar()
        for e in self.predicted_edges:
            if (e[0], e[1]) in self.removed_edges or (e[1], e[0]) in self.removed_edges:
                self.hit_result.append(1)
            else:
                self.hit_result.append(0)
        self.hit_result = np.array(self.hit_result)
        return self.hit_result

    def predict(self, u, v):
        return sum(1 / log(self.G.degree(w)) for w in nx.common_neighbors(self.G, u, v))

    def predict_weighted(self, u, v):
        return sum((self.G.edges[w, u]['weight'] + self.G.edges[w, v]['weight']) / log(self.G.degree(w)) for w in nx.common_neighbors(self.G, u, v))

    def adamic_adar(self):
        ebunch = nx.non_edges(self.G)
        if self.func == 'predict':
            temp_aa_predict = ((u, v, self.predict(u, v)) for u, v in ebunch)  
        elif self.func == 'predict_weighted':
            temp_aa_predict = ((u, v, self.predict_weighted(u, v)) for u, v in ebunch)  
        else:
            temp_aa_predict = ()
            logger.info('wrong func')
        return sorted(list(temp_aa_predict), key=lambda tup: tup[2], reverse=True)[:len(self.removed_edges)]

