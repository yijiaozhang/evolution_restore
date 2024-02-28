import numpy as np
from scipy.sparse.linalg import svds
import weightedlinkprediction.utils
logger = weightedlinkprediction.utils.get_logger(__name__)

class TSVDLinkPrediction(object):
    """
    TSVD Link prediction object
    Given a removed edges, collapsed_matrix and truncated_k, calculate hit@k.
    TSVD-CWT method in Reference: 
        Dunlavy, Daniel M., Tamara G. Kolda, and Evrim Acar. 
        "Temporal link prediction using matrix and tensor factorizations." 
        ACM Transactions on Knowledge Discovery from Data (TKDD) 5.2 (2011): 1-27.
    """

    def __init__(
        self,
        collapsed_matrix,
        remaining_edges,
        removed_edges,
        nodes_list,
        truncated_k = 10
    ):
        """
        :param collapsed_matrix: matrix of collapsed weighted tensor
        :param remaining_edges: list of remaining edges
        :param removed_edges: list of removed edges
        :param nodes_list: list of nodes
        :param truncated_k: the position k of truncated SVD
        """
        self.collapsed_matrix = collapsed_matrix
        self.remaining_edges = remaining_edges
        self.removed_edges = removed_edges
        self.nodes_list = nodes_list
        self.truncated_k = truncated_k

        self.ordered_edge_index = []
        self.hit_result = []

    def hit(self):
        self.ordered_edge_index = self.tsvd()
        temp = 0
        predicted_edges = []
        for e in self.ordered_edge_index[:len(self.remaining_edges) + len(self.removed_edges)]:
            if ((self.nodes_list[e[0]], self.nodes_list[e[1]]) not in self.remaining_edges) and ((self.nodes_list[e[1]], self.nodes_list[e[0]]) not in self.remaining_edges):
                predicted_edges.append((self.nodes_list[e[0]], self.nodes_list[e[1]]))
                temp += 1
                if temp == len(self.removed_edges):
                    break
        for e in predicted_edges:
            if ((e[0], e[1]) in self.removed_edges) or ((e[1], e[0]) in self.removed_edges):
                self.hit_result.append(1)
            else:
                self.hit_result.append(0)
        self.hit_result = np.array(self.hit_result)
        return self.hit_result
    
    def tsvd(self):
        u, s, vt = svds(self.collapsed_matrix, self.truncated_k)
        A_predict = u @ np.diag(s) @ vt
        A_predict *= np.tri(*A_predict.shape, k=-1)
        index_list = np.dstack(np.unravel_index(np.argsort(-A_predict.ravel()), (len(self.nodes_list), len(self.nodes_list))))
        index_list = index_list[0]
        index_list = index_list.tolist()
        return index_list
