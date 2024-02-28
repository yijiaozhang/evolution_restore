import numpy as np
import weightedlinkprediction.utils
logger = weightedlinkprediction.utils.get_logger(__name__)

class GenerateMatrix(object):
    """
    Generate Collasped  Matrix object
    Given a ordered edges and theta, generate the corresponding collasped matrix.
    """

    def __init__(
        self,
        nodes_list,
        theta = 0
    ):
        """
        :param nodes_list: list of nodes
        :param theta: a free parameter to tune the weight of time in collapsed matrix
        """
        self.nodes_list = nodes_list
        self.theta = theta

    def adjacency_matrix(self, edge_list):
        self.edge_list = edge_list
        adjacency_matrix = np.zeros((len(self.nodes_list), len(self.nodes_list)))
        for e in edge_list:
            adjacency_matrix[self.nodes_list.index(e[0])][self.nodes_list.index(e[1])] = 1
            adjacency_matrix[self.nodes_list.index(e[1])][self.nodes_list.index(e[0])] = 1
        return adjacency_matrix

    def matrix_by_orderedT(self, ordered_edge_list):
        self.ordered_edge_list = ordered_edge_list
        collapsed_matrix = np.zeros((len(self.nodes_list), len(self.nodes_list)))
        t_max = len(ordered_edge_list) + 1
        for i, e in enumerate(ordered_edge_list):
            t = i + 1
            collapsed_matrix[self.nodes_list.index(e[0])][self.nodes_list.index(e[1])] = (1 - self.theta)**(t_max - t)
            collapsed_matrix[self.nodes_list.index(e[1])][self.nodes_list.index(e[0])] = (1 - self.theta)**(t_max - t)
        return collapsed_matrix

    def matrix_by_realT(self, edge_df):
        self.edge_df = edge_df
        collapsed_matrix = np.zeros((len(self.nodes_list), len(self.nodes_list)))
        t_max = self.edge_df.real_time.max()
        for t in range(1, t_max+1):
            temp_df = self.edge_df.query('real_time == {}'.format(t))
            temp_edge_list = list(temp_df[['node1', 'node2']].itertuples(index=False, name=None))
            temp_matrix = np.zeros((len(self.nodes_list), len(self.nodes_list)))
            for e in temp_edge_list:
                temp_matrix[self.nodes_list.index(e[0])][self.nodes_list.index(e[1])] = (1 - self.theta)**(t_max - t)
                temp_matrix[self.nodes_list.index(e[1])][self.nodes_list.index(e[0])] = (1 - self.theta)**(t_max - t)
            collapsed_matrix += temp_matrix
        return collapsed_matrix