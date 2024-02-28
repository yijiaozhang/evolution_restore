import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
from getEdgeTime import *
from __Configuration import *


class Theoretic:
    def __init__(self):
        self.edge_nums = list(np.arange(100, 5000, 200))
        self.aucs = [round(i, 2) for i in np.arange(0.5, 0.91, 0.01)] + [round(i, 2) for i in np.arange(0.92, 1.01, 0.01)]
        self.dis = dict()


    def plot_average_distribute(self):
        edge_nums = [413, 6000, 2840, 598, 438, 2321]  # complex mc, fungi_mT, human_mt, fruit_fly_mT, worm_mT, bacteria_mT
        aucs = [0.76, 0.88, 0.89, 0.80, 0.76, 0.89]
        net_names = ["complex networks_maxconnect%"] + protein_net_names_mirrorTn  # protein_net_names
        times = 1000
        start = time.time()
        dis_mean = dict()
        dis_std = dict()
        for i in range(len(edge_nums)):
            auc = aucs[i]
            edge_num = edge_nums[i]
            net_name = net_names[i]
            path = "./judge_data/theoretic/" + net_name + "_" + str(edge_num) + "_" + str(auc) + "theoretic_average_distribute.txt"
            rerun = True
            if os.path.exists(path) and rerun == False:
                x = []
                y = []
                std = []
                with open(path, 'r') as f:
                    for line in f.readlines():
                        line = line.strip().split(' ')
                        x.append(float(line[0]))
                        y.append(float(line[1]))
                        std.append(float(line[2]))
            else:
                data = dict()
                for t in range(times):
                    edges = list(range(edge_num))
                    ep_judge = self.construct_ep_dict(edges, auc)
                    sorted_days = self.vote_sort(edges, ep_judge)
                    x, y = self.construct_diff_dis(edges, sorted_days)
                    for j in range(len(x)):
                        if x[j] in data.keys():
                            data[x[j]].append(y[j])
                        else:
                            data[x[j]] = [y[j]]
                    print("edge_num:", edge_num, "auc:", auc, "times", t, "done! time:", str(time.time() - start))
                    start = time.time()
                for d in data.keys():
                    data[d] = data[d] + [0] * (times - len(data[d]))
                    dis_mean[d] = np.mean(data[d])
                    dis_std[d] = np.std(data[d])
                mean = sorted(dis_mean.items(), key=lambda item: item[0])
                std = sorted(dis_std.items(), key=lambda item: item[0])
                x = [item[0] for item in mean]
                y = [item[1] for item in mean]
                std = [item[1] for item in std]
                self.save_mean_std(x, y, std, path)
            x = [x_i / edge_num for x_i in x]

    def plot_map_real_average_distribute(self):
        edge_nums = [413, 3249, 6000, 2840, 598, 438]  # complex mc, WTW, fungi_mT, human_mt, fruit_fly_mT, worm_mT,
        aucs = [0.76, 0.91, 0.88, 0.89, 0.80, 0.74]
        net_names = ["complex_networks%"] + ["WTW%"] + protein_net_names_mirrorTn  # protein_net_names
        random_num = 10
        start = time.time()
        for i in range(len(edge_nums)):
            auc = aucs[i]
            edge_num = edge_nums[i]
            net_name = net_names[i]
            path = "./judge_data/theoretic/" + net_name + "_" + str(edge_num) + "_" + str(auc) + "map_real_theoretic_average_distribute.txt"
            rerun = True
            if os.path.exists(path) and rerun == False:
                x = []
                y = []
                std = []
                with open(path, 'r') as f:
                    for line in f.readlines():
                        line = line.strip().split(' ')
                        x.append(float(line[0]))
                        y.append(float(line[1]))
                        std.append(float(line[2]))
            else:
                data = dict()
                edges = list(range(edge_num))
                ep_judge = self.construct_ep_dict(edges, auc)
                sorted_days = self.vote_sort(edges, ep_judge)
                days = get_descend_edge_time(net_name)
                original_days = []
                for day in days:
                    original_days.append(day[1])   # all real day
                x, y = self.map_and_cal_diff_dis(random_num, edges, sorted_days, original_days, path)
            x = [x_i / edge_num for x_i in x]
            print("net_name: ", net_name, "edge_num:", edge_num, "auc:", auc,  "done! time:", str(time.time() - start))

    def map_and_cal_diff_dis(self, random_num, edges, sort_days, original_days, path):
        days = set(original_days)
        data = dict()
        dis_mean = dict()
        dis_std = dict()
        for r in range(random_num):
            real_days = []
            tag = 0
            for day in days:
                index_location = []
                for i in original_days:
                    if i == day:
                        print(i)
                        index_location.append(tag)
                        tag += 1
                if len(index_location) != 1:
                    r_tmp = edges[index_location[0]:index_location[-1] + 1]  # 打乱对应real_days区间
                    random.shuffle(r_tmp)
                    for j in r_tmp:
                        real_days.append(j)
                else:
                    real_days.append(edges[index_location[0]])
            print(real_days)
            # calculate diffs
            diffs = dict()
            for i in range(len(real_days)):
                d_i = sort_days[i] - real_days[i]
                if d_i not in diffs.keys():
                    diffs[d_i] = 1
                else:
                    diffs[d_i] += 1
            for d_i in diffs.keys():
                diffs[d_i] = diffs[d_i] / len(real_days)
            diffs = sorted(diffs.items(), key=lambda item: item[0])
            x = [item[0] for item in diffs]
            y = [item[1] for item in diffs]
            for j in range(len(x)):
                if x[j] in data.keys():
                    data[x[j]].append(y[j])
                else:
                    data[x[j]] = [y[j]]
            print("times", r, "done!")
        for d in data.keys():
            data[d] = data[d] + [0] * (random_num - len(data[d]))
            dis_mean[d] = np.mean(data[d])
            dis_std[d] = np.std(data[d])
        mean = sorted(dis_mean.items(), key=lambda item: item[0])
        std = sorted(dis_std.items(), key=lambda item: item[0])
        x = [item[0] for item in mean]
        y = [item[1] for item in mean]
        std = [item[1] for item in std]
        self.save_mean_std(x, y, std, path)
        return x, y

    def save_mean_std(self, x, mean, std, path):
        with open(path, "w") as f:
            for i in range(len(x)):
                f.write(str(x[i]) + " " + str(mean[i]) + " " + str(std[i]) + "\n")

    def plot_somthing(self):
        exp_index = np.arange(-4.3, -2, 0.1)
        edge_nums = [int(1 / math.pow(10, e)) for e in exp_index]
        aucs = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        times = 10
        start = time.time()
        for auc in aucs:
            path = "./judge_data/theoretic/" + str(auc) + "_validate_something.txt"
            if os.path.exists(path):
                with open(path, 'r') as f:
                    plot_x = []
                    plot_y_mean = []
                    plot_y_std = []
                    for line in f.readlines():
                        line = line.strip().split(' ')
                        plot_x.append(float(line[0]))
                        plot_y_mean.append(float(line[1]))
                        plot_y_std.append(float(line[2]))
            else:
                with open(path, 'w') as f:
                    plot_x = [1 / e for e in edge_nums]
                    plot_y_mean = []
                    plot_y_std = []
                    for edge_num in edge_nums:
                        ys = []
                        for t in range(times):
                            edges = list(range(edge_num))
                            ep_judge = self.construct_ep_dict(edges, auc)
                            sorted_days = self.vote_sort(edges, ep_judge)
                            z = self.get_zindex(edges, sorted_days)
                            ys.append(z)
                            print("edge_num:", edge_num, "auc:", auc, "times:", t, "done! time:", str(time.time() - start))
                            start = time.time()
                        plot_y_mean.append(np.mean(ys))
                        plot_y_std.append(np.std(ys))
                    print(plot_x)
                    print(plot_y_mean)
                    for i in range(len(plot_y_mean)):
                        f.write(str(plot_x[i]) + ' ' + str(plot_y_mean[i]) + ' ' + str(plot_y_std[i]) + "\n")
            x = 1 - auc
            b = (1 - 2 * x) / math.sqrt(x * (1 - x))
            c = 0.7
            theoretic_y = [c / (b * math.pow(edge_num, 0.5)) for edge_num in edge_nums]
            plt.plot(plot_x, theoretic_y, ":", label="accu: " + str(auc) + "theoretic")
            print(auc, "Done!")


    def construct_diff_dis(self, real_days, sort_days):
        diffs = dict()
        for i in range(len(real_days)):
            d_i = sort_days[i] - real_days[i]
            if d_i not in diffs.keys():
                diffs[d_i] = 1
            else:
                diffs[d_i] += 1
        for d_i in diffs.keys():
            diffs[d_i] = diffs[d_i] / len(real_days)
        diffs = sorted(diffs.items(), key=lambda item: item[0])
        x = [item[0] for item in diffs]
        y = [item[1] for item in diffs]
        return x, y

    def vote_sort(self, edges, ep_judge):
        shuffle_edges_copy = list(edges)
        edge_score = dict(zip(shuffle_edges_copy, [0] * len(shuffle_edges_copy)))
        for edge1 in shuffle_edges_copy:
            _edge2s = shuffle_edges_copy[shuffle_edges_copy.index(edge1) + 1:]
            for edge2 in _edge2s:
                # 根据rule指标判断新旧
                if isinstance(ep_judge, dict):
                    judge = ep_judge[(edge1, edge2)]
                else:
                    judge = ep_judge(edge1, edge2)
                if judge == 1:
                    edge_score[edge1] += 1
                else:
                    edge_score[edge2] += 1
        sorted_edge_score_list = sorted(edge_score.items(), key=lambda item: item[1])
        sorted_edge_score = dict()
        for tuple_score in sorted_edge_score_list:
            sorted_edge_score[tuple_score[0]] = tuple_score[1]
        sorted_edge = list(sorted_edge_score.keys())
        days = [0] * sorted_edge.__len__()
        for edge in sorted_edge:
            sort_day = sorted_edge.index(edge)
            real_index = edges.index(edge)
            days[real_index] = sort_day
        return days

    def sa_sort(self, edges, ep_judge):
        shuffle_edges_copy = list(edges)
        edge_pair_judges = ep_judge
        print("SA sort prepared.")
        contradict_nums = []

        def random_reverse(edges):
            index = random.sample(list(range(len(edges))), 2)
            edges[index[0]], edges[index[1]] = edges[index[1]], edges[index[0]]
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


        tmp = 1e1
        tmp_min = 1e-20
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
                print("counter:", counter, "current contradict num:", s_old, "current temperature:", tmp, "calm_num:", calm_num)
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
        # shuffle_edges_copy
        days = [0] * shuffle_edges_copy.__len__()
        for edge in shuffle_edges_copy:
            sort_day = shuffle_edges_copy.index(edge)
            real_index = edges.index(edge)
            days[real_index] = sort_day
        print("sa sort finished!")
        return days

    def construct_ep_dict(self, edges, auc):
        if len(edges) < 10000:
            ep_judge = dict()
            for e1 in edges:
                for e2 in edges[edges.index(e1)+1:]:
                    if e1 != e2:
                        ep_judge[(e1, e2)] = e1 > e2
            wrong_num = int(len(ep_judge) * (1 - auc))
            wrong_keys = random.sample(list(ep_judge.keys()), wrong_num)
            for key in wrong_keys:
                ep_judge[key] = False if ep_judge[key] is True else True
            return ep_judge
        else:
            def ep_judge(e1, e2):
                if e1 < e2:
                    r = random.random()
                    if r < auc:
                        return False
                    else:
                        return True
                elif e1 > e2:
                    r = random.random()
                    if r < auc:
                        return True
                    else:
                        return False
            return ep_judge

    def get_zindex(self, edges, sorted_days) -> object:
        result_z = 0
        for i in range(len(edges)):
            d_i = abs(sorted_days[i] - edges[i])
            result_z += d_i
        result_z = (result_z / len(edges)) / len(edges)
        return result_z


if __name__ == '__main__':
    t = Theoretic()

    t.plot_average_distribute()
    # t.plot_somthing()
    # t.vote_vs_sa(auc=0.4, edge_num=1000)
    # t.plot_map_real_average_distribute()

