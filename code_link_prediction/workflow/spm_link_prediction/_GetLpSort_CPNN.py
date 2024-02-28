import numpy as np
import os
import scipy.io as scio
from multiprocessing import Pool
from getEdgeTime import get_all_edge_time
from getNet import get_mat, get_adj, divide_name
from _ShowModelAuc_ import get_all_k_accu
from _GetTSVD import get_keys
from __Configuration import *
import _EnsembleJudge_


def get_train_edges(net_name):
    edge_days = get_all_edge_time(net_name)
    T = max(edge_days.values())
    test_edges = []
    if net_name in "fungi_mirrorTn%4" + "human_mirrorTn%7" + "fruit_fly_mirrorTn%5":
        T = int(T)
        times = T - 2
        for t in np.arange(T, times, -1):
            if t == T:
                cur_edges = get_keys(edge_days, t)
                test_edges = test_edges + cur_edges
                print(net_name, ", max time:", T, ", cur deleted time:", t, ", cur edge nums in time", t, ": ",len(cur_edges), ", test edges num:", len(test_edges))
            else:
                cur_edges = get_keys(edge_days, t)
                test_edges = test_edges + cur_edges[0:int(len(cur_edges)/2)]
                print(net_name, ", max time:", T, ", cur deleted time:", t, ", cur edge nums in time", t, ": ", len(cur_edges), ", test edges num:", len(test_edges))
    elif net_name in "bacteria_mirrorTn%2":
        T = int(T)
        times = T - 1
        for t in np.arange(T, times, -1):
            if t == T:
                cur_edges = get_keys(edge_days, t)
                test_edges = test_edges + cur_edges[0:int(len(cur_edges) / 2)]
                print(net_name, ", max time:", T, ", cur deleted time:", t, ", cur edge nums in time", t, ": ", len(cur_edges), ", test edges num:", len(test_edges))
    elif net_name in "worm_mirrorTn%4":
        T = int(T)
        times = T - 2
        for t in np.arange(T, times, -1):
            cur_edges = get_keys(edge_days, t)
            test_edges = test_edges + cur_edges
            print(net_name, ", max time:", T, ", cur deleted time:", t, ", cur edge nums in time", t, ": ", len(cur_edges), ", test edges num:", len(test_edges))
    elif net_name in "weaver%" + "ants%" + "Air%" + "Coach%":
        T = int(T)
        times = T - 4
        for t in np.arange(T, times, -1):
            cur_edges = get_keys(edge_days, t)
            test_edges = test_edges + cur_edges
            print(net_name, ", max time:", T, ", cur deleted time:", t, ", cur edge nums in time", t, ": ", len(cur_edges), ", test edges num:", len(test_edges))
    elif net_name in "Ferry%":
        T = int(T)
        times = T - 3
        for t in np.arange(T, times, -1):
            cur_edges = get_keys(edge_days, t)
            test_edges = test_edges + cur_edges
            print(net_name, ", max time:", T, ", cur deleted time:", t, ", cur edge nums in time", t, ": ", len(cur_edges), ", test edges num:", len(test_edges))
    elif net_name in "WTW%":
        T = int(T)
        times = T - 14
        for t in np.arange(T, times, -1):
            cur_edges = get_keys(edge_days, t)
            test_edges = test_edges + cur_edges
            print(net_name, ", max time:", T, ", cur deleted time:", t, ", cur edge nums in time", t, ": ", len(cur_edges), ", test edges num:", len(test_edges))
    elif net_name in "fluctuations%" + "interfaces%" + "phase_transitions%" + "thermodynamics%" + "complex networks_maxconnect%" + "chaos_new%":
        times = int(T - T/3)
        for t in np.arange(T, times, -1):
            cur_edges = get_keys(edge_days, t)
            test_edges = test_edges + cur_edges
            print(net_name, ", max time:", T, ", cur deleted time:", t, ", cur edge nums in time", t, ": ", len(cur_edges), ", test edges num:", len(test_edges))
    edges = get_adj(net_name)
    train_edges = [k for k in edges if k not in test_edges]  # train edges
    return train_edges, test_edges


def get_net_data(net_name, net_name_lp, edges):
    """
    Return the adj of the network after removing the edges

    Returns the edge time of the network after removing the edges

    Returns the mat of the network after removing the edges
    """
    # return edge
    edge_path = "./data/spm_link_predition/" + net_name_lp + "/adj/"
    if os.path.exists(edge_path) is False:
        os.makedirs(edge_path)
    with open(edge_path + net_name_lp + ".txt", "w") as f:
        edgelist = []
        max_id = 0
        for line in edges:
            edge = str(line) + '\n'
            f.write(edge)
            edge_id = (int(line[0]), int(line[1]))
            if edge_id not in edgelist:
                edgelist.append(edge_id)
            cur_max = max(edge_id)
            max_id = cur_max if cur_max > max_id else max_id
    print("Get Net Adj")
    # return edge time
    all_edge_days = get_all_edge_time(net_name)
    edgetime = dict()
    for edge in edges:
        edgetime[edge] = all_edge_days[edge]
    time_path = "./data/spm_link_predition/" + net_name_lp + "/edgetime/"
    if os.path.exists(time_path) is False:
        os.makedirs(time_path)
    with open(time_path + net_name_lp + ".txt", "w") as f:
        for i in edgetime:
            i = str(i) + ":" + str(int(edgetime[i])) + '\n'
            f.write(i)
    print("Get Net Edgetime")
    # return mat
    mat_path = "./data/spm_link_predition/" + net_name_lp + "/mat/"
    if os.path.exists(mat_path) is False:
        os.makedirs(mat_path)
        mat = np.zeros((max_id + 1, max_id + 1))
        for edge_coord in edges:
            x = edge_coord[0]
            y = edge_coord[1]
            mat[x][y] = 1
            mat[y][x] = 1
        scio.savemat(mat_path + net_name_lp + ".mat", {'net': mat})
    else:
        mat = scio.loadmat(mat_path + net_name_lp + ".mat")['net']
    return net_name_lp


def get_edges_sort_time(net_name, edges, load_path, save_path, t):
    """
    input the adjacency matrix of the network and get the sorting time

    input network name, network edge, network adjacency matrix, return the sort time of edge

    edges : [(node1, node2), ...] , a tuple list of those sides represented by the upper triangular matrix of mat

    mat: Network adjacency matrix

    load_path: the loaded model path, None indicates the model trained with net_name

    load_net_name: net_name used by the loaded model.

    """

    shuffle_edges_copy = list(edges)
    edge_score = dict(zip(shuffle_edges_copy, [0] * len(shuffle_edges_copy)))
    judge_methods = [BEST_SINGLE, NODE2VEC_PAIR_NN, UNION_PAIR_NN, LINE_PAIR_NN, STRUC2VEC_PAIR_NN, DEEPWALK_PAIR_NN, SDNE_PAIR_NN]
    base_train_edges, test_edges, ensemble_train_edges = None, None, None
    TRAIN_EDGE_RATIO = 0.4
    if load_path is None:
        # load model
        load_path = "./model/ACCUbase" + str(7) + "_" + net_name + "_time" + str(t) + "_ratio" + str(TRAIN_EDGE_RATIO) + "/"
    else:
        pass
    ensemble = _EnsembleJudge_.EnsembleJudge(net_name, base_train_edges, test_edges, judge_methods, ensemble_train_edges,
                                             train_edge_ratio=TRAIN_EDGE_RATIO,
                                             save=False, save_path=None, load=True, load_path=load_path)
    model = ensemble
    print("Ensemble Model get!, sorting edges num :", len(shuffle_edges_copy))
    for edge1 in shuffle_edges_copy:
        _edge2s = shuffle_edges_copy[shuffle_edges_copy.index(edge1)+1:]
        for edge2 in _edge2s:
            judge = model.get_ep_judge((edge1, edge2), feature_dict_dict=None)
            if judge == 1:
                edge_score[edge1] += 1
                # print(edge1, edge_score[edge1])
            else:
                edge_score[edge2] += 1
        print((edge1), "total score: ", edge_score[edge1])
    # rank
    sorted_edge_score_list = sorted(edge_score.items(), key=lambda item: item[1])
    sorted_edge_score = dict()
    for tuple_score in sorted_edge_score_list:
        sorted_edge_score[tuple_score[0]] = tuple_score[1]
    sorted_edge = list(sorted_edge_score.keys())
    with open(save_path, 'w') as f:
        for edge in sorted_edge:
            sort_day = sorted_edge.index(edge)
            edge_days = str(edge[0]) + ' ' + str(edge[1]) + " " + str(sort_day + 1)
            f.write(edge_days + "\n")
        print("vote sort finished!")
    return edge_days


def test_main():
    times = 10
    net_names = ["fungi_mirrorTn%4"]
    ps = Pool(processes=5)
    for net_name in net_names:
        edges, test_edges = get_train_edges(net_name)
        net_name_, net_s = divide_name(net_name)
        net_name_lp = net_name_ + "_lp"
        mat_path = "./data/spm_link_predition/" + net_name_lp + "/"
        for t in range(1, times+1):
            save_path = "./sort/lp_svd/" + net_name_lp + "_time" + str(t) + ".txt"
            if os.path.exists(save_path) is False:
                model_load_path = "./model/ACCUbase" + str(7) + "_" + net_name_ + "_lp%" + "_time" + str(t) + "_ratio0.4/"
                if os.path.exists(model_load_path) is False:
                    get_net_data(net_name, net_name_lp, edges)  # save file
                    print(net_name_lp, "model", str(t), " begin!!")
                    get_all_k_accu(7, t, net_names=[net_name_lp + "%"],  force_calc=True, testornot=False, isTwoTrainSet=True)
                    load_net_name = net_name_lp + "%"
                    print(net_name_lp, str(t), "begin!!")
                    print(net_name_lp, get_edges_sort_time(load_net_name, edges, model_load_path, save_path, t))
                    print(net_name_lp, str(t), "commit!")
                else:
                    load_net_name = net_name_ + "_lp%"
                    print(load_net_name, str(t), "begin!!")
                    print(load_net_name, get_edges_sort_time(load_net_name, edges, model_load_path, save_path, t))
                    print(load_net_name, str(t), "commit!")
            else:
                print(save_path, "has done! or ", mat_path, "doesn't exist!")
    ps.close()
    ps.join()


if __name__ == '__main__':
    test_main()
