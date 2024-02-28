import random
import math
import numpy as np
import time
from getNet import get_adj
import getEdgeTime
from __Configuration import *


def get_all_edge_pairs(net_name):
    edges = get_adj(net_name)
    edge_days = getEdgeTime.get_all_edge_time(net_name)
    for edge1 in edges:
        _edge2s = edges[edges.index(edge1):]
        for edge2 in _edge2s:
            edge_pair = (edge1, edge2)
            new = getEdgeTime.is_different_time(edge_pair, edge_days)
            if new != -1:
                yield edge_pair, new


def get_random_edge_pairs(num, net_name):
    edges = get_adj(net_name)
    edge_days = getEdgeTime.get_all_edge_time(net_name)
    edge1_index = list(range(len(edges)))
    edge2_index = list(range(len(edges)))
    choosed_edge2_index_for_edge1_index = []
    for i in range(len(edges)):
        choosed_edge2_index_for_edge1_index.append([])
    for i in range(num):
        new = -1
        edge_pair = None
        while new == -1:
            edge1_i = random.choice(edge1_index)
            edge2_indexs = list(
                set(edge2_index[edge1_i:]) - {edge1_i} - set(choosed_edge2_index_for_edge1_index[edge1_i]))
            while edge2_indexs == []:
                edge1_index.remove(edge1_i)
                edge1_i = random.choice(edge1_index)
                edge2_indexs = list(
                    set(edge2_index[edge1_i:]) - {edge1_i} - set(choosed_edge2_index_for_edge1_index[edge1_i]))
            edge2_i = random.choice(edge2_indexs)
            edge_pair = (edges[edge1_i], edges[edge2_i])
            new = getEdgeTime.is_different_time(edge_pair, edge_days)
            choosed_edge2_index_for_edge1_index[edge1_i].append(edge2_i)
        yield edge_pair, new


def get_closure_edge_pairs(input_edge_pairs_dict, net_name):
    edge_index_dict = dict()  # {edge:0, edge:1, ...}
    edge_id = 0
    for edge_pair in input_edge_pairs_dict.keys():
        if edge_pair[0] not in edge_index_dict.keys():
            edge_index_dict[edge_pair[0]] = edge_id
            edge_id += 1
        if edge_pair[1] not in edge_index_dict.keys():
            edge_index_dict[edge_pair[1]] = edge_id
            edge_id += 1
    w = np.zeros((edge_id, edge_id))
    for edge_pair in input_edge_pairs_dict.keys():
        w[edge_index_dict[edge_pair[0]], edge_index_dict[edge_pair[1]]] = input_edge_pairs_dict[edge_pair]
        w[edge_index_dict[edge_pair[1]], edge_index_dict[edge_pair[0]]] = 1 - input_edge_pairs_dict[edge_pair]
    print("Matrix complete and begin warshall!! edge num:", edge_id)
    for k in range(edge_id):
        since = time.time()
        for i in range(edge_id):
            for j in range(edge_id):
                w[i, j] = int(w[i, j]) or (int(w[i, k]) and int(w[k, j]))
        print("Warshall k:" + str(k), " time:" + str(time.time() - since))
    print("Warshall done!!!!")
    input_edge_pairs = set(input_edge_pairs_dict.keys())
    ids = list(edge_index_dict.values())
    edges = list(edge_index_dict.keys())
    for i in range(edge_id):
        for j in range(i+1, edge_id):
            if w[i, j] == 1 or w[j, i] == 1:
                edge1 = edges[ids.index(i)]
                edge2 = edges[ids.index(j)]
                if (edge1, edge2) not in input_edge_pairs and (edge2, edge1) not in input_edge_pairs:
                    yield (edge1, edge2), w[i, j]


def get_train_test_edges(train_ratio, net_name):
    edges = get_adj(net_name)
    edge_days = getEdgeTime.get_all_edge_time(net_name)
    day_edges = dict()
    for edge in edge_days.keys():
        day = edge_days[edge]
        if day not in day_edges.keys():
            day_edges[day] = [edge]
        else:
            day_edges[day].append(edge)
    train_edges = []
    test_edges = []
    for day in day_edges.keys():
        temp_edge_list = day_edges[day]
        random.shuffle(temp_edge_list)
        train_num = math.ceil(len(temp_edge_list) * train_ratio)
        train_edges = train_edges + temp_edge_list[:train_num]
        test_edges = test_edges + temp_edge_list[train_num:]
    random.shuffle(train_edges)
    random.shuffle(test_edges)
    return train_edges, test_edges


def part_get_train_test_edges(train_ratio, net_name, time1, time2, ensemble_train_ratio=0.125):
    """
    :param train_ratio:
    :param net_name:
    :return: train_edges, test_edges
    """
    edge_days = getEdgeTime.get_all_edge_time(net_name)
    day_edges = dict()
    for edge in edge_days.keys():
        day = edge_days[edge]
        if int(day) == time1 or int(day) == time2:
            if day not in day_edges.keys():
                day_edges[day] = [edge]
            else:
                day_edges[day].append(edge)
    base_train_edges = []
    ensemble_train_edges = []
    test_edges = []
    for day in day_edges.keys():
        temp_edge_list = day_edges[day]
        random.shuffle(temp_edge_list)
        train_num = math.ceil(len(temp_edge_list) * train_ratio)
        ensemble_train_num = math.ceil(train_num * ensemble_train_ratio)
        base_train_edges = base_train_edges + temp_edge_list[:train_num-ensemble_train_num]
        ensemble_train_edges = ensemble_train_edges + temp_edge_list[train_num-ensemble_train_num:train_num]
        test_edges = test_edges + temp_edge_list[train_num:]
    random.shuffle(base_train_edges)
    random.shuffle(ensemble_train_edges)
    random.shuffle(test_edges)
    return base_train_edges, ensemble_train_edges, test_edges


def ensemble_get_train_test_edges(train_ratio, net_name, ensemble_train_ratio=0.125):
    if net_name not in ["chaos_new%"] + coauthor_net_names + ba_net_names + ba_model_nets + pso_net_names + fitness_net_names + coauthor_net_names_lp_svd:
        edge_days = getEdgeTime.get_all_edge_time(net_name)
        day_edges = dict()
        for edge in edge_days.keys():
            day = edge_days[edge]
            if day not in day_edges.keys():
                day_edges[day] = [edge]
            else:
                day_edges[day].append(edge)
        base_train_edges = []
        ensemble_train_edges = []
        test_edges = []
        for day in day_edges.keys():
            temp_edge_list = day_edges[day]
            random.shuffle(temp_edge_list)
            train_num = math.ceil(len(temp_edge_list) * train_ratio)
            ensemble_train_num = math.ceil(train_num * ensemble_train_ratio)
            b_index = train_num-ensemble_train_num if train_num-ensemble_train_num != 0 else 1
            base_train_edges = base_train_edges + temp_edge_list[:b_index]
            ensemble_train_edges = ensemble_train_edges + temp_edge_list[train_num-ensemble_train_num:train_num]
            test_edges = test_edges + temp_edge_list[train_num:]
        random.shuffle(base_train_edges)
        random.shuffle(ensemble_train_edges)
        random.shuffle(test_edges)
    else:
        edges = get_adj(net_name)
        random.shuffle(edges)
        train_num = math.ceil(len(edges) * train_ratio)
        ensemble_train_num = math.ceil(train_num * ensemble_train_ratio)
        b_index = train_num-ensemble_train_num
        base_train_edges = edges[:b_index]
        ensemble_train_edges = edges[b_index:train_num]
        test_edges = edges[train_num:]
    return base_train_edges, ensemble_train_edges, test_edges


def get_double_eps_from_edges(edges, net_name):
    edge_days = getEdgeTime.get_all_edge_time(net_name)
    for edge1 in edges:
        # _edge2s = edges[edges.index(edge1):]
        for edge2 in edges:
            if edge2 != edge1:
                edge_pair = (edge1, edge2)
                if type(edge1) is list or type(edge2) is list:
                    print("wrong place:", edge_pair)
                new = getEdgeTime.is_different_time(edge_pair, edge_days)
                if new != -1:
                    yield edge_pair, new


def get_single_eps_from_edges(edges, net_name, random_ratio=1) :
    edge_days = getEdgeTime.get_all_edge_time(net_name)
    for edge1 in edges:
        _edge2s = edges[edges.index(edge1):]
        for edge2 in _edge2s:
            if edge2 != edge1:
                if random.random() < random_ratio:
                    edge_pair = (edge1, edge2)
                    if type(edge1) is list or type(edge2) is list:
                        print("wrong place:", edge_pair)
                    new = getEdgeTime.is_different_time(edge_pair, edge_days)
                    if new != -1:
                        yield edge_pair, new


def get_single_eps_from_two_edges(train_edges, test_edges, net_name, random_ratio=1):
    edge_days = getEdgeTime.get_all_edge_time(net_name)
    for edge1 in train_edges:
        for edge2 in test_edges:
            if random.random() < random_ratio:
                edge_pair = (edge1, edge2)
                new = getEdgeTime.is_different_time(edge_pair, edge_days)
                if new != -1:
                    yield edge_pair, new


def get_single_eps_from_edges_label_by_timePair(edges, net_name, random_ratio=1):
    edge_days = getEdgeTime.get_all_edge_time(net_name)
    for edge1 in edges:
        _edge2s = edges[edges.index(edge1):]
        for edge2 in _edge2s:
            if edge2 != edge1:
                if random.random() < random_ratio:
                    edge_pair = (edge1, edge2)
                    if type(edge1) is list or type(edge2) is list:
                        print("wrong place:", edge_pair)
                    time_pair = (edge_days[edge1], edge_days[edge2])
                    if time_pair[0] < time_pair[1]:
                        yield edge_pair, time_pair
                    elif time_pair[1] < time_pair[0]:
                        yield (edge_pair[1], edge_pair[0]), (time_pair[1], time_pair[0])
                    else:
                        pass


