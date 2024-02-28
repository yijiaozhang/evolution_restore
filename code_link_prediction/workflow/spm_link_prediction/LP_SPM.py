import os
import copy
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from getEdgeTime import get_all_edge_time
from getNet import get_adj,  get_mat, divide_name


def get_delete_last_times(net_name):
    """
    Step 1:

    Removes the last moments of the original adjacency matrix

    return train mat after removing the edge
   """
    edge_days = get_all_edge_time(net_name)
    remain_edges = []
    if net_name in "fungi_mirrorTn%4" + "human_mirrorTn%7" + "fruit_fly_mirrorTn%5" + "bacteria_mirrorTn%2" + "Air%" + "Coach%" + "Ferry%":
        selected_t = [1, 2]
        for t in selected_t:
            if t == 1:
                cur_edges_t1 = get_keys(edge_days, t)
                remain_edges = remain_edges + cur_edges_t1
            elif t == 2:
                cur_edges_t2 = get_keys(edge_days, t)
                remain_edges = remain_edges + cur_edges_t2[0:int(len(cur_edges_t2) / 2)+1]
    elif net_name in "worm_mirrorTn%4":
        selected_t = [1, 2]
        for t in selected_t:
            cur_edges = get_keys(edge_days, t)
            remain_edges = remain_edges + cur_edges
    elif net_name in "weaver%":
        selected_t = [1, 2, 3, 4]
        for t in selected_t:
            cur_edges = get_keys(edge_days, t)
            remain_edges = remain_edges + cur_edges
    elif net_name in "ants%":
        selected_t = [1, 2, 3]
        for t in selected_t:
            cur_edges = get_keys(edge_days, t)
            remain_edges = remain_edges + cur_edges
    elif net_name in "WTW%":
        selected_t = [1997, 1998, 1999]
        for t in selected_t:
            cur_edges = get_keys(edge_days, t)
            remain_edges = remain_edges + cur_edges
    elif net_name in "fluctuations%" + "interfaces%" + "phase_transitions%" + "thermodynamics%" + "complex networks_maxconnect%" + "chaos_new%":
        max_times = max(edge_days.values())
        times = int(max_times - max_times / 3)
        selected_t = [i for i in range(1, times + 1)]
        for t in selected_t:
            cur_edges = get_keys(edge_days, t)
            remain_edges = remain_edges + cur_edges
    edges = get_adj(net_name)
    deleted_edges = [k for k in edges if k not in remain_edges]
    mat = get_mat(net_name)  # load net mat
    train_mat = copy.copy(mat)
    for i in deleted_edges:
        train_mat[i[0], i[1]] = 0
        train_mat[i[1], i[0]] = 0
    print(net_name, "train mat: ", (str(int(np.sum(train_mat)) / 2)),
          "remain edges: ", len(remain_edges), "deleted_edges: ", len(deleted_edges))
    return train_mat, remain_edges, deleted_edges


def get_keys(dic, value):
    return [key for key, v in dic.items() if v == value]


def get_spm(train_mat, remain_edges):
    """
    Step 2:

    input the deleted last-moment train mat into the SPM:

    1 train mat removes edges again according to the perturbation size (pertu_size) , and get the perturbation matrix (pertu_mat)
    2 The perturbation matrix is input into the perturbation algorithm to obtain the final probability matrix

    return:
       sim
    """
    probMatrix = np.zeros_like(train_mat)
    pertu_size = math.ceil(0.1 * (len(remain_edges)))
    perturbations = 30
    for pertus in range(0, perturbations):
        pertu_mat = copy.copy(train_mat)
        copy_edge = remain_edges
        random.shuffle(copy_edge)  # random
        pertu_edges = copy_edge[:pertu_size]
        for i in pertu_edges:
            pertu_mat[i[0], i[1]] = 0
            pertu_mat[i[1], i[0]] = 0
        probMatrix = probMatrix + perturbation(pertu_mat, train_mat)
    sim = probMatrix
    return sim


def perturbation(pertu_mat, train_mat):
    """
     Step 3: Input perturbation matrix (pertu_mat) and train mat

     return perturbed adjacency matrix:AdjTraining
    """
    # calculate eigenvalue eigenvector of train mat
    eigenvalue, eigenvector = np.linalg.eig(pertu_mat)
    ind1 = np.argsort(eigenvalue)  # rank
    eigenvalue = eigenvalue[ind1]
    eigenvector = eigenvector[:, ind1]  # column
    # find correct eigenvectors for perturbation of degenerate eigenvalues
    degenSign = np.zeros(len(eigenvalue), dtype="complex_")
    e_value = eigenvalue
    e_vector = eigenvector
    adjpertu = train_mat - pertu_mat
    for i in range(0, len(e_value)):
        if degenSign[i] == 0:
            x = abs((eigenvalue - eigenvalue[i]))
            index = np.argwhere(x < 10e-10)
            temp_eig = []
            if len(index) > 1:
                for j in index:
                    temp_eig.append(j[0])
                # print("temp_eig len: ", len(temp_eig), "index location: ", temp_eig)
                vRedun = eigenvector[:, temp_eig]
                cur_dot = np.dot(np.transpose(vRedun), adjpertu)
                m = np.dot(cur_dot, vRedun)
                m = (m + np.transpose(m))/2
                [e_val, e_vet] = np.linalg.eig(m)
                ind2 = np.argsort(e_val)
                e_val = e_val[ind2]
                e_vet = e_vet[:, ind2]  # column
                vRedun = np.dot(vRedun, e_vet)
                # renormalized the  new eigenvectors
                for j in range(0, len(m)):
                    vRedun[:, [j]] = vRedun[:, [j]] / np.linalg.norm(vRedun[:, [j]])
                e_vector[:, temp_eig] = vRedun  # update e_vector
                e_value[temp_eig] = eigenvalue[temp_eig] + e_val   # update e_value
                degenSign[temp_eig] = 1
    # perturbed the adjacency matrix AdjTraining
    arr1 = np.dot(np.transpose(e_vector), train_mat)
    arr2 = np.dot(arr1, e_vector)
    arr3 = np.diagonal(arr2)
    arr4 = np.dot(e_vector,  np.diag(arr3))
    AdjAnneal = np.dot(arr4, np.transpose(e_vector))
    return AdjAnneal


def count_hit_rank(sim, train_mat, deleted_edges):
    """
    Step 4:

    Count the hit sides
    """
    link_score = dict()
    spm_link = []
    non_link = np.where(np.triu(train_mat, 1) == 0)
    x = non_link[0]
    y = non_link[1]
    non_edges = [(x[i], y[i]) for i in range(x.size) if x[i] < y[i]]   # node1 < node2
    for i in non_edges:
        link_score[i] = sim[i[0]][i[1]]
    a = sorted(link_score.items(), key=lambda x: x[1], reverse=True)
    b = a[:len(deleted_edges)]
    for j in b:
        spm_link.append(j[0])
    hit_edges = [k for k in spm_link if k in deleted_edges]
    hit_ranks = []
    for i in hit_edges:
        index = spm_link.index(i)
        hit_ranks.append(index)
    return hit_edges, hit_ranks


def load_data(net_name, t):
    """
    step 1 in sort：

    load sort sequence after cpnn model
    """
    net_name_, net_t = divide_name(net_name)
    net_name_lp = net_name_ + "_lp"
    file_name = "./sort/lp_svd/" + net_name_lp + "_time" + str(t) + ".txt"
    edges_sort = {}
    with open(file_name, 'r') as fn:
        lines = fn.readlines()
        for line in lines:
            s1 = line.strip().strip("\n")
            s2 = s1.strip().split(' ')
            edges_sort[int(float(s2[0])), int(float(s2[1]))] = int(float(s2[2]))
    print("Get times ",  str(t), "vote  train edges sort dict----------------------------------- ")
    edges = get_adj(net_name)
    deleted_edges_sort = [k for k in edges if k not in edges_sort]  # deleted edges sort
    remain_edges_sort = [k for k in edges if k not in deleted_edges_sort]  # remain edges sort
    return edges_sort, remain_edges_sort, deleted_edges_sort


def collapsed_sort(net_name, theta, edges_sort, remain_edges_sort, deleted_edges_sort):
    """
    Step 2 in sort
    :
    The cumulative weight matrix is obtained
    """
    # according to T ,generate cur adj net
    T_sort = max(edges_sort.values())
    mat = get_mat(net_name)
    X_sort = np.zeros_like(mat)
    times = T_sort
    for t in range(1, times + 1):
        f = np.power((1 - theta), times - t)
        rank_mat_t = get_temporal_mat(net_name, edges_sort, t)
        print(net_name_lp, ": T=", times, ", t=", t, ", cur t=", t, " edge num : ", int(sum(np.sum(rank_mat_t, axis=1))/2), " Get mat T-t:", times - t, ", get f:", f)
        a = f * rank_mat_t
        X_sort = X_sort + a  # sum
    return X_sort, remain_edges_sort, deleted_edges_sort


def get_temporal_mat(net_name, edge_days, t):
    mat = get_mat(net_name)
    mat_t = np.zeros_like(mat)
    edges = get_keys(edge_days, t)
    for edge_coord in edges:
        x = edge_coord[0]
        y = edge_coord[1]
        mat_t[x][y] = 1
        mat_t[y][x] = 1
    return mat_t


def get_max_square_hit(net_name, t):
    """
    Step 3 in sort: Input data into the spm
    Step 4 in sort: Determine the theta value and the hit edge according to the area maximum of the lower part
    """
    edges_sort, remain_edges_sort, deleted_edges_sort = load_data(net_name, t)
    hit_edges = []
    hit_ranks = []
    hit_squares = []
    hit_thetas = []
    thetas = 0.003
    step = 0.0001
    for theta in np.arange(0.0001, thetas, step):
        train_mat_sort, remain_edges_sort, deleted_edges_sort = collapsed_sort(net_name, theta, edges_sort, remain_edges_sort, deleted_edges_sort)
        sim_sort = get_spm(train_mat_sort, remain_edges_sort)
        hit_edge, hit_rank = count_hit_rank(sim_sort, train_mat_sort, deleted_edges_sort)
        hit_square = cal_square(hit_rank, deleted_edges_sort)
        print("t=", t, "theta=", theta, "hit_square=", hit_square, "hit_num=", len(hit_edge))
        hit_edges.append(hit_edge)
        hit_ranks.append(hit_rank)
        hit_squares.append(hit_square)
        hit_thetas.append(theta)
    index = hit_squares.index(max(hit_squares))
    max_hit_edge = hit_edges[index]
    max_hit_rank = hit_ranks[index]
    max_hit_square = hit_squares[index]
    best_theta = hit_thetas[index]
    return max_hit_edge, max_hit_rank, max_hit_square, best_theta


def cal_square(hit_rank, deleted_edges):
    squrs = 0
    if len(hit_rank) != 0:
        for i in range(0, len(hit_rank)-1):
            sqr = (hit_rank[i+1]-hit_rank[i])*(i+1)
            squrs = squrs + np.sum(sqr)
        last_num = (len(deleted_edges)-hit_rank[-1])*len(hit_rank)
        # print("len(deleted_edges)", len(deleted_edges), "hit_rank",  hit_rank, "len(deleted_edges)", len(hit_rank))
        squrs = squrs + np.sum(last_num)
    else:
        squrs = 0
    return squrs


def measurement(hit_rank, deleted_edges):
    """
    measurement：The first i edges hit j edges
    returns: x,y for plot
    """
    x = range(0, len(deleted_edges))
    y = [0 for k in range(len(deleted_edges))]   # a list of all zeros as long as x
    for i in range(0, len(hit_rank)):
        if i + 1 == len(hit_rank):
            index1 = hit_rank[i]
            for j in range(index1, len(y)):
                y[j] = i + 1
        else:
            index1 = hit_rank[i]
            index2 = hit_rank[i + 1]
            for j in range(index1, index2):
                y[j] = i + 1
    return x, y


def plot_fig(net_name, x_real, y_real, x_sort, y_mean, y_std, times, change_ratio):
    # save
    fig_path = "./Result/sort_spm/" + net_name + "times" + str(times) + "_lp_spm.svg"
    save_path_real = "./Result/sort_spm/data/" + net_name + "times" + str(times) + "_lp_spm" + ".txt"
    with open(save_path_real, 'w') as f:
        for i, j in zip(list(range(len(x_real))), y_real):
            f.write(str(i) + ' ' + str(j) + "\n")
    save_path_sort = "./Result/sort_spm/data/" + net_name + "times" + str(times) + "_lp_spm_sort" + ".txt"
    with open(save_path_sort, 'w') as f:
        for i, j, k in zip(list(range(len(x_sort))), y_mean, y_std):
            f.write(str(i) + ' ' + str(j) + ' ' + str(k) + "\n")
    # plot
    fig = plt.figure(net_name)
    plt.plot(x_real, y_real, "#e8ac52", linewidth=2, linestyle='--')
    plt.plot(x_sort, y_mean, "#639aa0", linewidth=2)
    plt.fill_between(x_sort, y_mean-(1.96*y_std)/np.sqrt(times), y_mean+(1.96*y_std)/np.sqrt(times), color="#34b6c6", alpha=0.3)
    plt.grid(linestyle="-", c='lightgrey')
    if net_name in "WTW%":
        plt.legend(["SPM", "SPM with Restored Sequence"], frameon=False, fontsize=15, loc="lower right")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title([net_name, change_ratio])
    plt.xlabel('Rank of Predicted Links, r', fontsize=15)
    plt.ylabel('Number of Hits', fontsize=15)
    fig.savefig(fig_path, format='svg', dpi=150)


if __name__ == '__main__':
    net_names = ["fungi_mirrorTn%4",  "bacteria_mirrorTn%2", "fruit_fly_mirrorTn%5", "worm_mirrorTn%4", "WTW%",
    "weaver%", "ants%", "Air%", "Coach%", "Ferry%"
    "fluctuations%", "interfaces%", "phase_transitions%", "thermodynamics%","complex networks_maxconnect%","chaos_new%"]
    for net_name in net_names:
        # real
        train_mat, remain_edges, deleted_edges = get_delete_last_times(net_name)
        sim = get_spm(train_mat, remain_edges)
        hit_edges, hit_ranks = count_hit_rank(sim, train_mat, deleted_edges)
        hit_square = cal_square(hit_ranks, deleted_edges)
        x_real, y_real = measurement(hit_ranks, deleted_edges)
        print(net_name, "hit square:", hit_square, ", spm hit real edges num : ", len(hit_edges), ", hit edges:", hit_edges, ", hit ranks:", hit_ranks, )

        # sort
        times = 10
        results = []
        all_hit_ranks = []
        ps = Pool(processes=10)
        for t in range(1, times+1):
            # multi processing
            result = ps.apply_async(get_max_square_hit, args=(net_name, t))
            results.append(result)
        ps.close()
        ps.join()
        for res in results:
            max_hit_edge = res.get()[0]
            max_hit_rank = res.get()[1]
            max_hit_square = res.get()[2]
            best_theta = res.get()[3]
            print(net_name, ", max hit square:", max_hit_square, "best_theta =", best_theta, ", spm hit sort edges num : ", len(max_hit_edge), ", max hit edges:", max_hit_edge, ", max hit ranks:", max_hit_rank)
            x_sort, y_sort = measurement(max_hit_rank, deleted_edges)
            all_hit_ranks.append(y_sort)
            y_mean = np.mean(all_hit_ranks, axis=0)
            y_std = np.std(all_hit_ranks, axis=0)

        # plot
        change_ratio = (max_hit_square-hit_square)/hit_square
        print('change_ratio= ', change_ratio, "max_hit_square= ", max_hit_square, "hit_square= ", hit_square)
        plot_fig(net_name, x_real, y_real, x_sort, y_mean, y_std, times, change_ratio)


