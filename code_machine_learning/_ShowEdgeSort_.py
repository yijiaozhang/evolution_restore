import math
import os
import pickle
import random
import time
from multiprocessing import Manager, get_context
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LinearRegression
import _EnsembleJudge_
from __Configuration import *
from getEdgeTime import get_all_edge_time


class EdgeSort:

    def __init__(self, net_name, force_calc=False, withsa=False):
        # global TRAIN_EDGE_RATIO
        self.TRAIN_EDGE_RATIO = 0.4
        self.net_name = net_name
        self.withsa = withsa
        self.all_days = dict()
        file_name = "./sort/" + net_name + "_" + str(self.TRAIN_EDGE_RATIO) + "7_base_sort.pkl"
        if os.path.exists(file_name) is False or force_calc:
            self.base_train_edges = None
            self.ensemble_train_edges = None
            self.test_edges = None
            self.edges, self.real_days = self.get_sort_edges()
            self.shuffle_edges = list(self.edges)
            random.shuffle(self.shuffle_edges)
            # ensemble
            judge_methods = [BEST_SINGLE, NODE2VEC_PAIR_NN, UNION_PAIR_NN, LINE_PAIR_NN, STRUC2VEC_PAIR_NN,
                             DEEPWALK_PAIR_NN, SDNE_PAIR_NN]
            saveorload_path = "./model/ACCUbase" + str(7) + "_" + net_name + "_time" + str(0) + "_ratio" + str(self.TRAIN_EDGE_RATIO) + "/"
            ensemble = _EnsembleJudge_.EnsembleJudge(self.net_name, self.base_train_edges, self.test_edges,
                                                     judge_methods, self.ensemble_train_edges,
                                                     train_edge_ratio=self.TRAIN_EDGE_RATIO,
                                                     save=False, save_path=saveorload_path,
                                                     load=True, load_path=saveorload_path)
            self.model = ensemble
            self.random_days = self.random_sort()
            self.vote_sort_days = self.vote_sort()
            self.sa_sort_days = None if withsa is False else self.sa_sort()
            with open(file_name, 'wb') as file:
                self.all_days['real'] = self.real_days
                self.all_days['random'] = self.random_days
                self.all_days['ensemble_vote'] = self.vote_sort_days
                self.all_days['ensemble_sa'] = self.sa_sort_days
                self.all_days['edges'] = self.edges
                pickle.dump(self.all_days, file)
        else:
            with open(file_name, 'rb') as file:
                self.all_days = pickle.load(file)
                self.real_days = self.all_days['real']
                self.random_days = self.all_days['random']
                self.vote_sort_days = self.all_days['ensemble_vote']
                self.sa_sort_days = self.all_days['ensemble_sa']
                if self.sa_sort_days is None and withsa:
                    self.sa_sort()
                    with open(file_name, 'wb') as fw:
                        self.all_days['ensemble_sa'] = self.sa_sort_days
                        pickle.dump(self.all_days, fw)
                self.edges = self.all_days['edges']

    def get_sort_edges(self):
        edge_days = get_all_edge_time(self.net_name)
        edges = []
        days = []
        sorted_edge_days = sorted(set(edge_days.values()))
        for i in sorted_edge_days:
            i_day_edges = [edge for edge in edge_days.keys() if edge_days[edge] == i]
            print("i_day_edges:", i_day_edges.__len__())
            edges.extend(i_day_edges)
            days.extend([sorted_edge_days.index(i)] * i_day_edges.__len__())
        return edges, days

    def random_sort(self):
        shuffle_edges_copy = list(self.shuffle_edges)
        edges_l = shuffle_edges_copy.__len__()
        for i in range(edges_l):
            for j in range(i):
                x = random.random()
                if x < 0.5:
                    new_edge_judge = 1
                else:
                    new_edge_judge = -1
                if new_edge_judge > 0:
                    temp = shuffle_edges_copy[i]
                    shuffle_edges_copy[j + 1:i + 1] = shuffle_edges_copy[j:i]
                    shuffle_edges_copy[j] = temp
                    break
                else:
                    pass
        days = [0] * edges_l
        for edge in shuffle_edges_copy:
            sort_day = shuffle_edges_copy.index(edge)
            real_index = self.edges.index(edge)
            days[real_index] = sort_day
        print("random sort finished!")
        return days

    def get_score(edge1, edge2, i, model, file_name, lock):
        try:
            judge = model.get_ep_judge((edge1, edge2))
            lock.acquire()
            with open(file_name, 'rb') as f:
                edge_score = pickle.load(f)
                if judge == 1:
                    edge_score[edge1] += 1
                else:
                    edge_score[edge2] += 1
            with open(file_name, 'wb') as f:
                pickle.dump(edge_score, f)
            print(i, "edge pair done!")
            lock.release()
        except Exception as ex:
            print(ex)

    def vote_sort(self):
        shuffle_edges_copy = list(self.shuffle_edges)
        edge_score = dict(zip(shuffle_edges_copy, [0] * len(shuffle_edges_copy)))
        model = self.model
        start = time.time()
        lock = Manager().Lock()
        temp_file_name = "./sort/temp_score/score.pkl"
        with get_context("spawn").Pool(processes=6) as ps:
            i = 0
            for edge1 in shuffle_edges_copy:
                _edge2s = shuffle_edges_copy[shuffle_edges_copy.index(edge1):]
                print("current edge num", shuffle_edges_copy.index(edge1))
                for edge2 in _edge2s:
                    # single processing
                    judge = model.get_ep_judge((edge1, edge2))
                    if judge == 1:
                        edge_score[edge1] += 1
                    else:
                        edge_score[edge2] += 1
                    # (multi processing)
        #             i += 1
        #             ps.apply_async(self.get_score,args=(edge1, edge2, i, model, temp_file_name, lock, ))
        #     ps.close()
        #     ps.join()
        # with open(temp_file_name, 'wb') as f:
        #     pickle.dump(edge_score, f)
        print(self.net_name, "vote sort.", "score time:", str(time.time() - start))
        start = time.time()
        sorted_edge_score_list = sorted(edge_score.items(), key=lambda item: item[1])
        print(self.net_name, "vote sort.", "sort time:", str(time.time() - start))
        sorted_edge_score = dict()
        for tuple_score in sorted_edge_score_list:
            sorted_edge_score[tuple_score[0]] = tuple_score[1]
        sorted_edge = list(sorted_edge_score.keys())
        days = [0] * sorted_edge.__len__()
        for edge in sorted_edge:
            sort_day = sorted_edge.index(edge)
            real_index = self.edges.index(edge)
            days[real_index] = sort_day
        print("vote sort finished!")
        return days

    def sa_sort(self):
        shuffle_edges_copy = list(self.shuffle_edges)
        model = self.model
        edge_pair_judges = dict()
        for e1 in shuffle_edges_copy:
            for e2 in shuffle_edges_copy[shuffle_edges_copy.index(e1):]:
                if e2 != e1:
                    edge_pair_judges[(e1, e2)] = model.get_ep_judge((e1, e2))
        print("SA sort prepared.")
        contradict_nums = []

        def random_reverse(edges):
            index = random.sample(list(range(len(edges))), 2)
            edges[index[0]], edges[index[1]] = edges[index[1]], edges[index[0]]
            # edges[0], edges[1] = edges[1], edges[0]
            pass

        def get_contra_num(edges):
            contra_num = 0
            for e1 in edges:
                for e2 in edges[edges.index(e1):]:
                    if e2 != e1:
                        if (e1, e2) in edge_pair_judges.keys() and edge_pair_judges[(e1, e2)] == 1:
                            contra_num += 1
                        elif (e2, e1) in edge_pair_judges.keys() and edge_pair_judges[(e2, e1)] == 0:
                            contra_num += 1
                        else:
                            pass
            return contra_num

        def judge(dE, tmp):
            if dE < 0:
                return True
            else:
                d = math.exp(-((dE) / tmp))
                if d > random.random():
                    return True
                else:
                    return False

        def plot_contradict_nums(nums):
            x = list(range(len(nums)))
            y = nums
            plt.figure()
            plt.xticks([i * 10 for i in x], fontsize=10)
            plt.yticks(fontsize=10)
            plt.xlabel("iter num", fontsize=15, fontweight='bold')
            plt.ylabel("contradict num", fontsize=15, fontweight='bold')
            plt.plot(x, y, '-*', color='r')
            plt.savefig("./Result/sort/" + self.net_name + "sa_sort_middle.png", dpi=600)
            plt.show()
        tmp = 1e1
        tmp_min = 1e-15
        alpha = 0.999
        s_old = get_contra_num(shuffle_edges_copy)
        counter = 0
        calm_num = 0
        while tmp > tmp_min:
            after_reverse_edges = list(shuffle_edges_copy)
            random_reverse(after_reverse_edges)
            s_new = get_contra_num(after_reverse_edges)
            dE = s_new - s_old
            if judge(dE, tmp):
                s_old = s_new
                shuffle_edges_copy = after_reverse_edges
            counter += 1
            if counter % 100 == 0:
                contradict_nums.append(s_old)
                print("counter:", counter, "current contradict num:", s_old, "current temperature:", tmp)
                if counter > 300 and s_old >= contradict_nums[-2] - 10:
                    calm_num += 1
                if calm_num >= 10:
                    print("Break! counter:", counter, "current contradict num:", s_old, "current temperature:", tmp)
                    break
            if dE >= 0:
                tmp = tmp * alpha
            if counter > 100000 or s_old < 10 or tmp < tmp_min:
                print("Break! counter:", counter, "current contradict num:", s_old, "current temperature:", tmp)
                break
        days = [0] * shuffle_edges_copy.__len__()
        for edge in shuffle_edges_copy:
            sort_day = shuffle_edges_copy.index(edge)
            real_index = self.edges.index(edge)
            days[real_index] = sort_day
        print("sa sort finished!")
        return days

    def save_ensemble_sort(self):
        """

        :rtype: object
        """
        real_days = np.array(self.real_days)
        vote_sort_days = np.array(self.vote_sort_days)
        file_name = "./sort/" + self.net_name + '_' + str(
            self.TRAIN_EDGE_RATIO) + "model1_7_base_vote_ensemble_sort.txt"
        sort_edge_days = dict()
        with open(file_name, 'w') as f:
            for edge, i in zip(self.edges, list(range(len(self.edges)))):
                sort_edge_days[edge] = vote_sort_days[i]
                f.write(str(edge[0]) + ' ' + str(edge[1]) + " " + str(vote_sort_days[i] + 1) + "\n")  # 持久化
        if self.sa_sort_days is not None:
            sa_sort_days = np.array(self.sa_sort_days)
            file_name = "./sort/" + self.net_name + '_' + str(
                self.TRAIN_EDGE_RATIO) + "model1_7_base_sa_ensemble_sort.txt"
            sort_edge_days = dict()
            with open(file_name, 'w') as f:
                for edge, i in zip(self.edges, list(range(len(self.edges)))):
                    sort_edge_days[edge] = sa_sort_days[i]
                    f.write(str(edge[0]) + ' ' + str(edge[1]) + " " + str(sa_sort_days[i] + 1) + "\n")  # 持久化
        print(self.net_name + "ensemble save Done!")


    def kendall_tau(_sorted, _real):
        diff_time_ep_num = 0
        right_ep_num = 0
        wrong_ep_num = 0
        for edge1_index in range(len(_real)):
            for edge2_index in range(edge1_index, len(_real)):
                if _real[edge1_index] != _real[edge2_index]:
                    diff_time_ep_num += 1
                    if _sorted[edge2_index] != _sorted[edge1_index] and (_real[edge2_index] > _real[edge1_index]) == (
                            _sorted[edge2_index] > _sorted[edge1_index]):
                        right_ep_num += 1
                    else:
                        wrong_ep_num += 1
        return round((right_ep_num - wrong_ep_num) / diff_time_ep_num, 3)

    def spearman_footrule(_sorted, _real):
        dis = 0
        for i in range(len(_real)):
            dis += abs(_real[i] - _sorted[i])
        return dis

    def set_based(_sorted, _real):
        dis = 0
        for i in range(len(_real)):
            real_set = set(_real[:i + 1])
            sorted_set = set(_sorted[:i + 1])
            join_set = real_set & sorted_set
            dis += len(join_set) / (i + 1)
        return round(dis / len(_real), 3)

    def get_node_degree_from_edges(edges):
        """
        :param edges:
        :return:
        """
        node_degree = dict()
        for edge in edges:
            if edge[0] not in node_degree.keys():
                node_degree[edge[0]] = 1
            else:
                node_degree[edge[0]] += 1
            if edge[1] not in node_degree.keys():
                node_degree[edge[1]] = 1
            else:
                node_degree[edge[1]] += 1
        return node_degree


def main():
    # if you want to use this code, you must build the net's model at first (you can use get_all_accu to build model)
    # get_all_accu(7, net_names=net_names, times=10)
    net_names = ["fruit_fly_mirrorTn%5"]
    for net_name in net_names:
        edge_sort = EdgeSort(net_name, force_calc=True, withsa=False)
        edge_sort.save_ensemble_sort()


if __name__ == '__main__':
    main()
