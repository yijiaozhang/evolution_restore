import numpy as np
import pandas as pd
import sys
import weightedlinkprediction.utils
import weightedlinkprediction.generate_collapsed_matrix
import weightedlinkprediction.tsvd_predict
logger = weightedlinkprediction.utils.get_logger(__name__)

if __name__ == '__main__':
    real_time_path = sys.argv[1]
    vote_time_path = sys.argv[2]
    truncated_k = int(sys.argv[3])
    theta = float(sys.argv[4])
    network_name = sys.argv[5]
    repeat = int(sys.argv[6])
    output_path = sys.argv[-1]
    network_name_list = []
    repeat_list = []
    k_list = []
    theta_list = []
    hit_realT_list = []
    hit_restoredT_list = []
    real_edge_time_df = pd.read_csv(real_time_path, sep=' ', names=['node1', 'node2', 'real_time'])
    vote_edge_time_df = pd.read_csv(vote_time_path, sep=' ', names=['node1', 'node2', 'vote_time'])
    nodes_list = list(pd.unique(real_edge_time_df[['node1', 'node2']].values.ravel('K')))
    real_edge_time_df = real_edge_time_df.merge(vote_edge_time_df, on=['node1', 'node2'], how='left', indicator=True)
    removed_edges_df = real_edge_time_df[real_edge_time_df._merge == 'left_only']
    removed_edge_list = list(removed_edges_df[['node1', 'node2']].itertuples(index=False, name=None))
    edge_df = real_edge_time_df[real_edge_time_df._merge == 'both']
    edge_df = edge_df.sort_values('vote_time') # order edge_list by vote timestamp
    edge_list = list(edge_df[['node1', 'node2']].itertuples(index=False, name=None))
    logger.info("Working on network {}, theta = {}, k = {}".format(network_name, theta, truncated_k))
    collapsed_matrix_real = weightedlinkprediction.generate_collapsed_matrix.GenerateMatrix(nodes_list, theta).matrix_by_realT(edge_df)
    collapsed_matrix_vote = weightedlinkprediction.generate_collapsed_matrix.GenerateMatrix(nodes_list, theta).matrix_by_orderedT(edge_list)
    ## 1. calculate hitted number by real T
    temp_hit_real = weightedlinkprediction.tsvd_predict.TSVDLinkPrediction(collapsed_matrix_real, edge_list, removed_edge_list, nodes_list, truncated_k).hit()
    hit_realT_list.append(list(temp_hit_real))
    ## 2. calculate hitted number by vote T
    temp_hit_vote = weightedlinkprediction.tsvd_predict.TSVDLinkPrediction(collapsed_matrix_vote, edge_list, removed_edge_list, nodes_list, truncated_k).hit()
    hit_restoredT_list.append(list(temp_hit_vote))
    network_name_list.append(network_name)
    k_list.append(truncated_k)
    theta_list.append(theta)
    repeat_list.append(repeat)
    result_df = pd.DataFrame(
        {'network_name':network_name_list, 'theta':theta_list, 'k':k_list, 'counter':repeat_list, 
        'hit_realT':hit_realT_list, 'hit_restoredT':hit_restoredT_list}
        )
    result_df.to_csv(output_path, index=None)
