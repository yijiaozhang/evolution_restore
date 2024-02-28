import os
from getNet import get_adj
from getDict import get_dict
from getNet import divide_name
from __Configuration import *


def get_all_edge_time(net_name):
    """
    :rtype: object
    :param net_name:
    :return: {(node1, node2): edgetime, ...}
    """
    net_type, net_time = divide_name(net_name)
    if net_type in protein_net_types_mirrorTn:
        file_name = './net_data/' + net_type + "/edgetime/" + str(net_time) + '_edges_day.txt'
        edge_days = dict()
        had_file = os.path.exists(file_name)
        if had_file is False:
            edge_list = get_adj(net_name)
            adjs = dict()
            with open(file_name, 'w') as f:
                for edge in edge_list:
                    edge_day = 0
                    for day in range(1, int(net_time)+1):
                        if day not in adjs.keys():
                            adj = get_adj(net_type + "%" + str(day))
                            adjs[day] = adj
                        else:
                            adj = adjs[day]
                        if edge_day == 0 and edge in adj:
                            edge_day = day
                            break
                    edge_days[edge] = edge_day
                    f.write(str(edge) + ":" + str(edge_day) + "\n")
        else:
            edge_days = get_dict(file_name)
        return edge_days
    elif net_type == "weaver":
        f_name = "./net_data/" + net_type + "/edgetime/" + net_type + ".txt"
        edge_days = get_dict(f_name)
        return edge_days
    elif net_type == "ants":
        f_name = "./net_data/" + net_type + "/edgetime/" + net_type + ".txt"
        edge_days = get_dict(f_name)
        return edge_days
    elif net_name in transport_net_names:
        f_name = "./net_data/" + net_type + "/edgetime/" + net_type + ".txt"
        edge_days = get_dict(f_name)
        return edge_days
    elif net_name in ba_net_names:
        f_name = "./net_data/" + net_type + "/edgetime/" + net_type + ".txt"
        edge_days = get_dict(f_name)
        return edge_days
    elif net_name in ba_model_nets:
        f_name = "./net_data/" + net_type + "/edgetime/" + net_type + ".txt"
        edge_days = get_dict(f_name)
        return edge_days
    elif net_name in pso_net_names:
        f_name = "./net_data/" + net_type + "/edgetime/" + net_type + ".txt"
        edge_days = get_dict(f_name)
        return edge_days
    elif net_name in fitness_net_names:
        f_name = "./net_data/" + net_type + "/edgetime/" + net_type + ".txt"
        edge_days = get_dict(f_name)
        return edge_days
    elif net_name in coauthor_net_names_max_connect:
        f_name = "./net_data/" + net_type + "/edgetime/" + net_type + ".txt"
        edge_days = get_dict(f_name)
        return edge_days
    elif net_type == "chaos_new":
        f_name = "./net_data/" + net_type + "/edgetime/" + net_type + ".txt"
        edge_days = get_dict(f_name)
        return edge_days
    elif net_name in coauthor_net_names:
        f_name = "./net_data/" + net_type + "/edgetime/" + net_type + ".txt"
        edge_days = get_dict(f_name)
        return edge_days
    elif net_name in economy_net_names:
        f_name = "./net_data/" + net_type + "/edgetime/" + net_type + ".txt"
        edge_days = get_dict(f_name)
        return edge_days
    # lp_svd
    elif net_name in coauthor_net_names_lp_svd:
        f_name = "./net_data/lp_svd/" + net_type + "/edgetime/" + net_type + ".txt"
        edge_days = get_dict(f_name)
        return edge_days
    elif net_name in protein_net_names_mirrorTn_lp_svd:
        f_name = "./net_data/lp_svd/" + net_type + "/edgetime/" + net_type + ".txt"
        edge_days = get_dict(f_name)
        return edge_days
    elif net_name in interaction_net_names_lp_svd:
        f_name = "./net_data/lp_svd/" + net_type + "/edgetime/" + net_type + ".txt"
        edge_days = get_dict(f_name)
        return edge_days
    elif net_name in transport_net_names_lp_svd:
        f_name = "./net_data/lp_svd/" + net_type + "/edgetime/" + net_type + ".txt"
        edge_days = get_dict(f_name)
        return edge_days
    elif net_name in economy_net_names_lp_svd:
        f_name = "./net_data/lp_svd/" + net_type + "/edgetime/" + net_type + ".txt"
        edge_days = get_dict(f_name)
        return edge_days


def is_different_time(edge_pair, edge_days) :
    """

    :param edge_pair: ((node1, node2), (node3, node4))
    :param edge_days: get_all_edge_day(day)
    :return:

    """
    edge1, edge2 = edge_pair[0], edge_pair[1]
    if edge1 in edge_days.keys() and edge2 in edge_days.keys():
        day1 = edge_days[edge1]
        day2 = edge_days[edge2]
        if day1 < day2:
            new = 0
        elif day1 > day2:
            new = 1
        else:
            new = -1
        return new
    else:
        return -1


def get_sorted_edge_time(edge_days):
    """
    :param edge_days:
    :return:
    """
    days = set(edge_days.values())
    sorted_days = sorted(days)
    for edge in edge_days:
        edge_days[edge] = sorted_days.index(edge_days[edge])
    return edge_days


def get_descend_edge_time(net_name):
    """
    :param edge_days:
    :return:
    """
    edge_days = get_all_edge_time(net_name)
    edge_sorted_days = sorted(edge_days.items(), key=lambda x: x[1])
    return edge_sorted_days



if __name__ == '__main__':
    # _edge_days_ = get_all_edge_time("sci_net_contacts%51")
    # get_all_edge_time('ht09_contacts%')
    # get_all_edge_time("ant%")
    # get_all_edge_time("fungi%4")
    # get_all_edge_time("human%7")
    edge_days = get_all_edge_time("complex_networks%")
    # get_all_edge_time("worm%4")
    # get_all_edge_time("bacteria%2")
    # get_all_edge_time("facebook%")
    # get_all_edge_time("coauthor%")
    get_descend_edge_time(edge_days)
