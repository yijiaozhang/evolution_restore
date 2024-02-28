import numpy as np
import pandas as pd
import networkx as nx
import sys
import weightedlinkprediction.utils
import weightedlinkprediction.adamic_adar_predict
logger = weightedlinkprediction.utils.get_logger(__name__)

if __name__ == '__main__':
    real_time_path = sys.argv[1]
    vote_time_path = sys.argv[2]
    network_name = sys.argv[3]
    output_path = sys.argv[-1]
    real_edge_time_df = pd.read_csv(real_time_path, sep=' ', names=['node1', 'node2', 'real_time'])
    vote_edge_time_df = pd.read_csv(vote_time_path, sep=' ', names=['node1', 'node2', 'vote_time'])
    nodes_list = list(pd.unique(real_edge_time_df[['node1', 'node2']].values.ravel('K')))
    real_edge_time_df = real_edge_time_df.merge(vote_edge_time_df, on=['node1', 'node2'], how='left', indicator=True)
    removed_edges_df = real_edge_time_df[real_edge_time_df._merge == 'left_only']
    removed_edge_list = list(removed_edges_df[['node1', 'node2']].itertuples(index=False, name=None))
    edge_list = list(vote_edge_time_df[['node1', 'node2']].itertuples(index=False, name=None))   
    logger.info("Working on network {}".format(network_name))
    G = nx.Graph()
    G.add_nodes_from(nodes_list)
    G.add_edges_from(edge_list)
    hit_result = weightedlinkprediction.adamic_adar_predict.AdamicAdar(G, 'predict', removed_edge_list).hit()
    result_df = pd.DataFrame(
        {'network_name':[network_name], 'hit_adj':[list(hit_result)]}
        )   
    result_df.to_csv(output_path, index=None)
