import os
import networkx as nx
import pickle
import time
import math
import numpy as np
import scipy.sparse as sp
from getDict import get_dict
from getNet import get_adj, get_nx
from getEdgeTime import get_all_edge_time


def get_feature_dict(net_mat, feature_name, net_name, force_calc=False):
    """
    feature names
    :param net_mat:
    :param feature_name:
    :param net_name:
    :param force_calc
    :return:
    """
    if feature_name == "bn":
        return get_betweenness_dict(net_mat, net_name, force_calc)
    elif feature_name == 'cn':
        return get_cn_dict(net_mat, net_name, force_calc)
    elif feature_name == "degree":
        return get_degree_dict(net_mat, net_name, force_calc)
    elif feature_name == "strength":
        return get_strength_dict(net_mat, net_name, force_calc)
    elif feature_name == "cc":
        return get_clustering_coefficient_dict(net_mat, net_name, force_calc)
    elif feature_name == "ra":
        return get_resource_alloc_dict(net_mat, net_name, force_calc)
    elif feature_name == "aa":
        return get_adamic_adar_dict(net_mat, net_name, force_calc)
    elif feature_name == "pa":
        return get_pa_dict(net_mat, net_name, force_calc)
    elif feature_name == "lp":
        return get_local_path_dict(net_mat, net_name, force_calc)
    elif feature_name == 'k_shell':
        return get_k_shell_dict(net_mat, net_name, force_calc)
    elif feature_name == 'pr':
        return get_pagerank_dict(net_mat, net_name, force_calc)
    elif feature_name == '-log_bn':
        file_name = './edge_betweeness/' + str(net_name) + 'edge_betweeness.pickle'
        with open(file_name, 'rb') as file:
            feature_dict = pickle.load(file)
        for edge in feature_dict.keys():
            feature_dict[edge] = -math.log(feature_dict[edge])
        return feature_dict
    elif feature_name == 'interval':
        return get_all_edge_time(net_name)
    else:
        return None


def get_pagerank_dict(net_mat, net_name, force_calc=False):
    file_name = './edge_prs/' + str(net_name) + 'edges_prs.txt'
    if os.path.exists(file_name) is False or force_calc:
        nx_net = get_nx(net_name)
        print("Node pagerank getting...")
        pr = nx.pagerank(nx_net, weight=None)
        print("Node pagerank get!")
        edges = get_adj(net_name)
        edge_prs = dict()
        with open(file_name, 'w') as file:
            for edge in edges:
                edge_prs[edge] = max(np.real(pr[edge[0]]), np.real(pr[edge[1]]))
                file.write(str(edge) + ':' + str(edge_prs[edge]) + '\n')
    else:
        edge_prs = get_dict(file_name)
    return edge_prs


def get_betweenness_dict(net_mat, net_name, force_calc=False):
    file_name = './edge_betweeness/' + str(net_name) + 'edge_betweeness.pickle'
    had_file = os.path.exists(file_name)
    if had_file is False or force_calc:
        net = get_nx(net_name)
        betweenness_edge = nx.edge_betweenness_centrality(net)
        with open(file_name, 'wb') as file:
            pickle.dump(betweenness_edge, file)
    else:
        with open(file_name, 'rb') as file:
            betweenness_edge = pickle.load(file)
    return betweenness_edge


def get_cn_dict(net_mat, net_name, force_calc=False):
    file_name = './edge_cns/' + str(net_name) + 'edges_cns.txt'
    if os.path.exists(file_name) is False or force_calc:
        net = get_nx(net_name)
        edge_cns = dict()
        edges = get_adj(net_name)
        with open(file_name, 'w') as file:
            edge_num = 0
            since = time.time()
            for edge in edges:
                edge_num += 1
                neighbor1, neighbor2 = set(net.neighbors(edge[0])), set(net.neighbors(edge[1]))
                edge_cns[edge] = len(neighbor1 & neighbor2)
                file.write(str(edge) + ':' + str(edge_cns[edge]) + '\n')
                if edge_num % 100 == 0:
                    since = time.time()
    else:
        edge_cns = get_dict(file_name)
    return edge_cns


def get_degree_dict(net_mat, net_name, force_calc=False):
    file_name = './edge_degrees/' + str(net_name) + 'edges_degrees.txt'
    if os.path.exists(file_name) is False or force_calc:
        net = get_nx(net_name)
        node_degrees = net.degree()
        edge_degrees = dict()
        edges = get_adj(net_name)
        with open(file_name, 'w') as file:
            edge_num = 0
            since = time.time()
            for edge in edges:
                edge_num += 1
                degree = node_degrees[int(edge[0])] + node_degrees[int(edge[1])]
                edge_degrees[edge] = degree
                file.write(str(edge) + ':' + str(degree) + '\n')
                if edge_num % 100 == 0:
                    since = time.time()
    else:
        edge_degrees = get_dict(file_name)
    return edge_degrees


def get_strength_dict(net_mat, net_name, force_calc=False):
    """
   The Strength of weak lies
    strength(Jaccard)
    :param net_mat:
    :param net_name:
    :return:
    """
    file_name = './edge_strengths/' + str(net_name) + 'edges_strengths.txt'
    if os.path.exists(file_name) is False or force_calc:
        net = get_nx(net_name)
        edge_strengths = dict()
        edges = get_adj(net_name)
        with open(file_name, 'w') as file:
            edge_num = 0
            since = time.time()
            for edge in edges:
                edge_num += 1
                neighbor1, neighbor2 = set(net.neighbors(edge[0])), set(net.neighbors(edge[1]))
                strength = len(neighbor1 & neighbor2) / len(neighbor1 | neighbor2)

                edge_strengths[edge] = strength
                file.write(str(edge) + ':' + str(strength) + '\n')
                if edge_num % 100 == 0:
                    since = time.time()
    else:
        edge_strengths = get_dict(file_name)
    return edge_strengths


def get_clustering_coefficient_dict(net_mat, net_name, force_calc=False):
    """
    cc(HPI)
    :param net_mat:
    :param net_name:
    :return:
    """
    file_name = './edge_ccs/' + str(net_name) + 'edge_ccs.txt'
    if os.path.exists(file_name) is False or force_calc:
        net = get_nx(net_name)
        edge_ccs = dict()
        edges = get_adj(net_name)
        with open(file_name, 'w') as file:
            edge_num = 0
            since = time.time()
            for edge in edges:
                edge_num += 1
                neighbor1, neighbor2 = set(net.neighbors(edge[0])), set(net.neighbors(edge[1]))
                min_len1_len2 = min(len(neighbor1) - 1, len(neighbor2) - 1)
                if min_len1_len2 == 0:
                    cc = 0
                else:
                    cc = len(neighbor1 & neighbor2) / min_len1_len2

                edge_ccs[edge] = cc
                file.write(str(edge) + ':' + str(cc) + '\n')
                if edge_num % 100 == 0:
                    since = time.time()
    else:
        edge_ccs = get_dict(file_name)
    return edge_ccs


def get_resource_alloc_dict(net_mat, net_name, force_calc=False):
    file_name = './edge_ras/' + str(net_name) + 'edge_ras.txt'
    if os.path.exists(file_name) is False or force_calc:
        net = get_nx(net_name)
        edge_ras = dict()
        edges = get_adj(net_name)
        with open(file_name, 'w') as file:
            edge_num = 0
            since = time.time()
            for edge in edges:
                edge_num += 1
                neighbor1, neighbor2 = set(net.neighbors(edge[0])), set(net.neighbors(edge[1]))
                common_neighbor = neighbor1 & neighbor2
                ra = 0
                for node in common_neighbor:
                    ra += 1 / net.degree[node]
                edge_ras[edge] = ra
                file.write(str(edge) + ':' + str(ra) + '\n')
                if edge_num % 100 == 0:
                    since = time.time()
    else:
        edge_ras = get_dict(file_name)
    return edge_ras


def get_adamic_adar_dict(net_mat, net_name, force_calc=False):
    file_name = './edge_aas/' + str(net_name) + 'edge_aas.txt'
    if os.path.exists(file_name) is False or force_calc:
        net = get_nx(net_name)
        edge_aas = dict()
        edges = get_adj(net_name)
        with open(file_name, 'w') as file:
            edge_num = 0
            since = time.time()
            for edge in edges:
                edge_num += 1
                neighbor1, neighbor2 = set(net.neighbors(edge[0])), set(net.neighbors(edge[1]))
                common_neighbor = neighbor1 & neighbor2
                aa = 0
                for node in common_neighbor:
                    aa += 1 / math.log(net.degree[node])
                edge_aas[edge] = aa
                file.write(str(edge) + ':' + str(aa) + '\n')
                if edge_num % 100 == 0:
                    since = time.time()
    else:
        edge_aas = get_dict(file_name)
    return edge_aas


def get_pa_dict(net_mat, net_name, force_calc=False):
    file_name = './edge_pas/' + str(net_name) + 'edge_pas.txt'
    if os.path.exists(file_name) is False or force_calc:
        net = get_nx(net_name)
        edge_pas = dict()
        edges = get_adj(net_name)
        with open(file_name, 'w') as file:
            edge_num = 0
            since = time.time()
            for edge in edges:
                edge_num += 1
                neighbor1, neighbor2 = set(net.neighbors(edge[0])), set(net.neighbors(edge[1]))
                pa = len(neighbor1) * len(neighbor2)
                edge_pas[edge] = pa
                file.write(str(edge) + ':' + str(pa) + '\n')
                if edge_num % 100 == 0:
                    since = time.time()
    else:
        edge_pas = get_dict(file_name)
    return edge_pas


def get_local_path_dict(net_mat, net_name, force_calc=False):
    file_name = './edge_lps/' + str(net_name) + 'edge_lps.txt'
    if os.path.exists(file_name) is False or force_calc:
        edge_lps = dict()
        edges = get_adj(net_name)
        alpha = 0.01
        if isinstance(net_mat, np.ndarray):
            s = np.linalg.matrix_power(net_mat, 2) + alpha * np.linalg.matrix_power(net_mat, 3)
        elif isinstance(net_mat, sp.coo_matrix):
            s = net_mat.dot(net_mat) + alpha * net_mat.dot(net_mat.dot(net_mat))
            s = sp.dok_matrix(s)
        else:
            s = None
        with open(file_name, 'w') as file:
            edge_num = 0
            since = time.time()
            for edge in edges:
                edge_num += 1

                lp = s[edge[0], edge[1]]

                edge_lps[edge] = lp
                file.write(str(edge) + ':' + str(lp) + '\n')
                if edge_num % 100 == 0:
                    since = time.time()
    else:
        edge_lps = get_dict(file_name)
    return edge_lps


def get_k_shell_dict(net_mat, net_name, force_calc=False):
    file_name = './edge_k_shells/' + str(net_name) + 'edge_k_shells.txt'
    if os.path.exists(file_name) is False or force_calc:
        net = get_nx(net_name)
        edges = get_adj(net_name)
        edge_k_shells = dict()

        k = 0
        while len(net.nodes()) != 0:
            node_num_in_cur_step = 1
            edge_num_in_cur_k = 0
            while node_num_in_cur_step != 0:
                node_num_in_cur_step = 0
                node_degrees = net.degree()
                cur_nodes = list(net.nodes())
                for node in cur_nodes:
                    if node_degrees[node] > k:
                        pass
                    else:
                        for neighbor in net[node]:
                            if (node, neighbor) in edges:
                                edge_k_shells[(node, neighbor)] = k
                            else:
                                edge_k_shells[(neighbor, node)] = k
                            edge_num_in_cur_k += 1
                        node_num_in_cur_step += 1
                        net.remove_node(node)
            print("k_shell:", k, "finished.", "edge_num:", edge_num_in_cur_k, "node_num:", len(net.nodes()),
                  "edge_num:", len(net.edges()))
            k += 1
        with open(file_name, 'w') as file:
            for edge in edges:
                file.write(str(edge) + ':' + str(edge_k_shells[edge]) + '\n')
    else:
        edge_k_shells = get_dict(file_name)
    return edge_k_shells


def judge_by_feature(feature_name, feature_dict, edge_pair):
    feature1, feature2 = feature_dict[edge_pair[0]], feature_dict[edge_pair[1]]
    if feature1 > feature2:
        return 0
    elif feature1 == feature2:
        return -1
    else:
        return 1
