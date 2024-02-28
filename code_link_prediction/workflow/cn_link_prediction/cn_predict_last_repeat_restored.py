import numpy as np
import pandas as pd
import networkx as nx
import sys
import weightedlinkprediction.utils
import weightedlinkprediction.common_neighber_predict
logger = weightedlinkprediction.utils.get_logger(__name__)

if __name__ == '__main__':
    real_time_path = sys.argv[1]
    vote_time_path = sys.argv[2]
    theta = float(sys.argv[3])
    network_name = sys.argv[4]
    repeat = int(sys.argv[5])
    output_path = sys.argv[-1]
    real_edge_time_df = pd.read_csv(real_time_path, sep=' ', names=['node1', 'node2', 'real_time'])
    vote_edge_time_df = pd.read_csv(vote_time_path, sep=' ', names=['node1', 'node2', 'vote_time'])
    nodes_list = list(pd.unique(real_edge_time_df[['node1', 'node2']].values.ravel('K')))
    real_edge_time_df = real_edge_time_df.merge(vote_edge_time_df, on=['node1', 'node2'], how='left', indicator=True)
    removed_edges_df = real_edge_time_df[real_edge_time_df._merge == 'left_only']
    removed_edge_list = list(removed_edges_df[['node1', 'node2']].itertuples(index=False, name=None))
    vote_edge_time_df['weight'] = (1 - theta)**(vote_edge_time_df.vote_time.max() - vote_edge_time_df.vote_time) 
    weighted_edge_list = list(vote_edge_time_df[['node1', 'node2', 'weight']].itertuples(index=False, name=None))   
    logger.info("Working on network {}, theta = {}, repeat = {}".format(network_name, theta, repeat))
    G = nx.Graph()
    G.add_nodes_from(nodes_list)
    G.add_weighted_edges_from(weighted_edge_list)
    hit_result = weightedlinkprediction.common_neighber_predict.CommonNeighber(G, 'predict_weighted', removed_edge_list).hit()
    result_df = pd.DataFrame(
        {'network_name':[network_name], 'theta':[theta], 'counter':[repeat],  'hit_restoredT':[list(hit_result)]}
        )   
    result_df.to_csv(output_path, index=None)
