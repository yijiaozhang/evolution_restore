import scipy.io as scio
import scipy
import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
import json
import time
import pickle
import h5py
import copy
from __Configuration import *


def divide_name(net_name: object) -> object:
    """

    :rtype: object
    """
    _net_name_ = net_name.split('%')
    return _net_name_[0], _net_name_[1]


def get_mat(net_name):
    net_type, net_time = divide_name(net_name)
    if net_type in protein_net_types_mirrorTn:
        f_name = "./net_data/" + net_type + "/mat/" + net_time + ".mat"
        if os.path.exists(f_name) is False:
            if net_type == 'fungi_mirrorTn':
                ori_file_name = "./net_data/" + net_type + "/ori_adj/" + "4932_out.T" + str(net_time) + ".AgainAdj_mirrorTn.txt"
            elif net_type == 'human_mirrorTn':
                ori_file_name = "./net_data/" + net_type + "/ori_adj/" + "9696_out.T" + str(net_time) + ".AgainAdj_mirrorTn.txt"
            elif net_type == 'fruit_fly_mirrorTn':
                ori_file_name = "./net_data/" + net_type + "/ori_adj/" + "7227_out.T" + str(net_time) + ".AgainAdj_mirrorTn.txt"
            elif net_type == 'worm_mirrorTn':
                ori_file_name = "./net_data/" + net_type + "/ori_adj/" + "6239_out.T" + str(net_time) + ".AgainAdj_mirrorTn.txt"
            elif net_type == 'bacteria_mirrorTn':
                ori_file_name = "./net_data/" + net_type + "/ori_adj/" + "83333_out.T" + str(net_time) + ".AgainAdj_mirrorTn.txt"
            else:
                ori_file_name = None
            with open(ori_file_name, 'r') as f:
                edgelist = []
                max_id = 0
                for line in f:
                    line = line.strip().split('\t')
                    edge = (int(line[0])-1, int(line[1])-1)
                    if edge not in edgelist:
                        edgelist.append(edge)
                    cur_max = max(edge)
                    max_id = cur_max if cur_max > max_id else max_id
                mat = np.zeros((max_id + 1, max_id + 1))
                for edge in edgelist:
                    if edge[0] != edge[1]:
                        mat[edge] = 1
                        mat[(edge[1], edge[0])] = 1
                # mat = sp.coo_matrix(mat)
                scio.savemat(f_name, {'net': mat})
        else:
            mat = scio.loadmat(f_name)['net']
        return mat
    elif net_type in "Air%" + "Coach%" + "Ferry%":
        f_name = "./net_data/" + net_type + "/mat/" + net_type + ".mat"
        if os.path.exists(f_name) is False:
           f_list = "./net_data/" + net_type + "/adj/" + net_type + ".txt"
           with open(f_list, 'r') as f:
               edgelist = []
               max_id = 0
               for line in f:
                    line = line.strip().split('\t')
                    line = line[0]
                    line = line.strip("()")
                    line = line.strip().split(",")
                    edge = (int(line[0]), int(line[1]))
                    if edge not in edgelist:
                        edgelist.append(edge)
                    cur_max = max(edge)
                    max_id = cur_max if cur_max > max_id else max_id
               mat = np.zeros((max_id + 1, max_id + 1))
           for edge_coord in edgelist:
                x = edge_coord[0]
                y = edge_coord[1]
                mat[x][y] = 1
                mat[y][x] = 1
           scio.savemat(f_name, {'net': mat})
        else:
            mat = scio.loadmat(f_name)['net']
        return mat
    elif net_type == "weaver":
        f_name = "./net_data/" + net_type + "/mat/" + net_type + ".mat"
        mat = scio.loadmat(f_name)['net']
        return mat
    elif net_type == "ants":
        f_name = "./net_data/" + net_type + "/mat/" + net_type + ".mat"
        mat = scio.loadmat(f_name)['net']
        return mat
    elif net_type == "employee":
        f_name = "./net_data/" + net_type + "/mat/" + net_type + ".mat"
        mat = scio.loadmat(f_name)['net']
        return mat
    elif net_name in transport_net_names:
        f_name = "./net_data/" + net_type + "/mat/" + net_type + ".npz"
        mat = scipy.sparse.load_npz(f_name)
        return mat
    elif net_name in ba_net_names:
        f_name = "./net_data/" + net_type + "/mat/" + net_type + ".mat"
        mat = h5py.File(f_name, 'r')[net_type]
        mat = np.transpose(mat)
        return mat
    elif net_name in ba_model_nets:
        f_name = "./net_data/" + net_type + "/mat/" + net_type + ".mat"
        mat = h5py.File(f_name, 'r')['net']
        mat = np.transpose(mat)
        return mat
    elif net_name in pso_net_names:
        f_name = "./net_data/" + net_type + "/mat/" + net_type + ".mat"
        mat = h5py.File(f_name, 'r')['net']
        mat = np.transpose(mat)
        return mat
    elif net_name in fitness_net_names:
        f_name = "./net_data/" + net_type + "/mat/" + net_type + ".mat"
        mat = h5py.File(f_name, 'r')['net']
        mat = np.transpose(mat)
        return mat
    elif net_name in gbg_net_names:
        f_name = "./net_data/" + net_type + "/mat/" + net_type + ".mat"
        mat = scio.loadmat(f_name)['net']
        return mat
    elif net_name in coauthor_net_names_max_connect:
        f_name = "./net_data/" + net_type + "/mat/" + net_type + ".mat"
        mat = h5py.File(f_name, 'r')['net']
        mat = np.transpose(mat)
        return mat
    elif net_name in coauthor_net_names:
        f_name = "./net_data/" + net_type + "/mat/" + net_type + ".mat"
        mat = h5py.File(f_name, 'r')['net']
        mat = np.transpose(mat)
        return mat
    elif net_name in economy_net_names:
        f_name = "./net_data/" + net_type + "/mat/" + net_type + ".mat"
        mat = h5py.File(f_name, 'r')['net']
        mat = np.transpose(mat)
        return mat
    elif net_name in coauthor_net_names_lp_svd:
        f_name = "./net_data/lp_svd/" + net_type + "/mat/" + net_type + ".mat"
        mat = scio.loadmat(f_name)['net']
        return mat
    elif net_name in protein_net_names_mirrorTn_lp_svd:
        f_name = "./net_data/lp_svd/" + net_type + "/mat/" + net_type + ".mat"
        mat = scio.loadmat(f_name)['net']
        return mat
    elif net_name in interaction_net_names_lp_svd:
        f_name = "./net_data/lp_svd/" + net_type + "/mat/" + net_type + ".mat"
        mat = scio.loadmat(f_name)['net']
        return mat
    elif net_name in transport_net_names_lp_svd:
        f_name = "./net_data/lp_svd/" + net_type + "/mat/" + net_type + ".mat"
        mat = scio.loadmat(f_name)['net']
        return mat
    elif net_name in economy_net_names_lp_svd:
        f_name = "./net_data/lp_svd/" + net_type + "/mat/" + net_type + ".mat"
        mat = scio.loadmat(f_name)['net']
        return mat


def get_adj(net_name):
    """
    :param net_name:
    :return: [(node1, node2), ...]
    """
    net_type, net_time = divide_name(net_name)
    if net_time != "":
        f_name = "./net_data/" + net_type + "/adj/" + net_time + ".txt"
    elif net_time == "" and net_type + "%" in coauthor_net_names_lp_svd + protein_net_names_mirrorTn_lp_svd + interaction_net_names_lp_svd + transport_net_names_lp_svd + economy_net_names_lp_svd:
        f_name = "./net_data/lp_svd/" + net_type + "/adj/" + net_type + ".txt"
    else:
        f_name = "./net_data/" + net_type + "/adj/" + net_type + ".txt"

    had_file = os.path.exists(f_name)
    if had_file is False:
        mat = get_mat(net_name)
        if isinstance(mat, np.ndarray):
            edge_coord = np.where(np.triu(mat) == 1)
        elif isinstance(mat, sp.coo_matrix):
            edge_coord = sp.find(sp.triu(mat))
        else:
            edge_coord = None
        x = edge_coord[0]
        y = edge_coord[1]
        edges = []
        for j in range(len(x)):
            t = (x[j], y[j])
            edges.append(t)
        with open(f_name, 'w') as f:
            for edge in edges:
                f.write(str(edge) + "\n")
        print("get_adj: ", net_time, "edge_num: ", edges.__len__())
    else:
        edges = []
        with open(f_name, 'r') as f:
            for line in f:
                edges.append(tuple(eval(line.strip())))
    return edges


def get_nx(net_name):
    """
    get Graph（networkx）
    :rtype: object
    :param net_name:
    :return:
    """
    mat = get_mat(net_name)
    if isinstance(mat, np.ndarray):
        g = nx.from_numpy_array(mat)
    elif isinstance(mat, sp.coo_matrix):
        g = nx.from_scipy_sparse_matrix(mat)
    else:
        g = None
    return g


def get_node(mat):
    # mat = get_mat(net_name)
    if isinstance(mat, np.ndarray):
        g = nx.from_numpy_array(mat)
    elif isinstance(mat, sp.coo_matrix):
        g = nx.from_scipy_sparse_matrix(mat)
    else:
        g = None
    edge_node = g.nodes
    nodes = []
    for node in edge_node._nodes:
        nodes.append(node)
    return nodes


def get_net_sort_degree(net_name):
    """

    :rtype: object
    """

    adj_mat = get_mat(net_name)
    if np.count_nonzero(adj_mat[0]) == 0:
        a = np.delete(adj_mat, 0, 0)
        b = np.delete(a, 0, 1)
        mat = b
    else:
        mat = adj_mat
    if isinstance(mat, np.ndarray):
        g = nx.from_numpy_array(mat)
    elif isinstance(mat, sp.coo_matrix):
        g = nx.from_scipy_sparse_matrix(mat)
    else:
        g = None
    node_degree = sorted(g.degree(), key=lambda x: x[1], reverse=True)
    node_degrees = copy.copy(node_degree)
    for k in node_degrees:
        if k[1] == 0:
            node_degree.remove(k)
    nodes_sort = []
    for node in node_degree:
        nodes_sort.append(node[0])
    nodes = []
    for i in nodes_sort:
        nodes.append(i + 1)
    if np.count_nonzero(adj_mat[0]) == 0:
        return nodes
    else:
        return nodes_sort


if __name__ == '__main__':
    get_adj('worm_mirrorTn%4')



