import time
import os
import xlwt
import numpy as np
import pickle
from multiprocessing import Manager,  get_context
import getEdgePair
from getNet import get_adj
from __Configuration import *
from _EnsembleJudge_ import EnsembleJudge
from _SingleJudge_ import SingleJudge
from getEdgeTime import get_all_edge_time


def get_train_test_data(net_name, times, train_edge_ratio):
    if os.path.exists("./judge_data/different_ratio/datas/") is False:
        os.makedirs("./judge_data/different_ratio/datas/")
    file_name = "./judge_data/different_ratio/datas/" + net_name + "_times" + str(times) + "_ratio" + str(train_edge_ratio) + ".pkl"
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            base_train_edges, ensemble_train_edges, test_edges = pickle.load(f)
    else:
        base_train_edges, ensemble_train_edges, test_edges = getEdgePair.ensemble_get_train_test_edges(train_edge_ratio, net_name)
        with open(file_name, 'wb') as f:
            pickle.dump((base_train_edges, ensemble_train_edges, test_edges), f)
    return base_train_edges, ensemble_train_edges, test_edges


def single_ratio_process(net_name, ratio_ranges, TRAIN_EDGE_RATIO, judge_methods, auc_data, file_name, lock=None):
    try:
        print("Process", net_name, TRAIN_EDGE_RATIO)
        times = 20 if TRAIN_EDGE_RATIO < 0.1 else 10
        for t in range(times):
            data_file_name = "./judge_data/different_ratio/datas/" + net_name + "_times" + str(t) + "_ratio" + str(TRAIN_EDGE_RATIO) + ".pkl"
            model_file_name = "./model/DIFFRATIO" + "_" + net_name + "_time" + str(t) + "_ratio" + str(TRAIN_EDGE_RATIO) + "/Ensemble/weights.pkl"
            ratio_pos = np.where(ratio_ranges == TRAIN_EDGE_RATIO)[0][0]
            if TRAIN_EDGE_RATIO < 0.3 and np.sum(auc_data[t][ratio_pos, :]) != 0 and os.path.exists(data_file_name) and os.path.exists(model_file_name):
                print("NET:", net_name, "Times:", t, "Train_edge_ratio:", TRAIN_EDGE_RATIO, " End.")
                with open("./log/" + str(net_name) + "diff_log.txt", 'w') as log_f:
                    log_f.write(
                        "NET:" + net_name + "Times:" + str(t) + "Train_edge_ratio:" + str(TRAIN_EDGE_RATIO) + " End.\n")
                continue
            else:
                since = time.time()
                print("NET:", net_name, "Times:", t, "Train_edge_ratio:", TRAIN_EDGE_RATIO, " Begin...")
                base_train_edges, ensemble_train_edges, test_edges = get_train_test_data(net_name, t, TRAIN_EDGE_RATIO)
                saveorload_path = "./model/DIFFRATIO" + "_" + net_name + "_time" + str(t) + "_ratio" + str(TRAIN_EDGE_RATIO) + "/"
                if net_name == "fungi%4":
                    test_ep_ratio = 0.01
                elif net_name in ["fungi%4", "human%7"] and TRAIN_EDGE_RATIO < 0.1:
                    test_ep_ratio = 0.01
                else:
                    test_ep_ratio = 1
                model = EnsembleJudge(net_name, base_train_edges, test_edges, judge_methods, ensemble_train_edges,
                                      train_edge_ratio=TRAIN_EDGE_RATIO,
                                      test_ep_ratio=test_ep_ratio,
                                      save=True, save_path=saveorload_path,
                                      load=True, load_path=saveorload_path)
                test_auc = model.get_test_auc_two_train_edges_ensemble_stacking()
                single_judge = SingleJudge(net_name, base_train_edges, test_edges, ensemble_train_edges)
                _, single_is_big_olds, single_test_auc = single_judge.all_feature_judge()
                if lock is None:
                    auc_data[t][ratio_pos, :] = list(test_auc.values()) + list(single_test_auc.values())
                    print("NET:", net_name, "Times:", t, "Train_edge_ratio:", TRAIN_EDGE_RATIO,
                          " End. Time consume is " + str(time.time() - since))
                    with open("./log/" + str(net_name) + "diff_log.txt", 'w') as log_f:
                        log_f.write("NET:" + net_name + "Times:" + str(t) + "Train_edge_ratio:" + str(
                            TRAIN_EDGE_RATIO) + " End. Time consume is " + str(time.time() - since) + "\n")
                    with open(file_name, 'wb') as f:
                        pickle.dump(auc_data, f)
                else:
                    lock.acquire()
                    with open(file_name, 'rb') as f:
                        auc_data = pickle.load(f)
                    auc_data[t][ratio_pos, :] = list(test_auc.values()) + list(single_test_auc.values())
                    print("NET:", net_name, "Times:", t, "Train_edge_ratio:", TRAIN_EDGE_RATIO,
                          " End. Time consume is " + str(time.time() - since))
                    with open("./log/" + str(net_name) + "diff_log.txt", 'w') as log_f:
                        log_f.write("NET:" + net_name + "Times:" + str(t) + "Train_edge_ratio:" + str(
                            TRAIN_EDGE_RATIO) + " End. Time consume is " + str(time.time() - since) + "\n")
                    with open(file_name, 'wb') as f:
                        pickle.dump(auc_data, f)
                    lock.release()
    except Exception as ex:
        print(ex)


def get_different_edge_ratio_auc(net_name, multi_ratio_process=False):
    # ensemble test
    print(net_name, "Begin...")
    os.system("taskset -p 0xff %d" % os.getpid())
    # global TRAIN_EDGE_RATIO
    TRAIN_EDGE_RATIO = 0.1
    judge_methods = [BEST_SINGLE, NODE2VEC_PAIR_NN, UNION_PAIR_NN, LINE_PAIR_NN, STRUC2VEC_PAIR_NN, DEEPWALK_PAIR_NN, SDNE_PAIR_NN]
    times = 20
    auc_l = len(judge_methods) + 1 + len(EDGE_FEATURE)
    ratio_ranges = np.round(list(np.arange(0.01, 0.1, 0.01)) + list(np.arange(0.1, 1, 0.1)), 3)

    excel_f = xlwt.Workbook()
    show_data_auc_mean = np.zeros((len(ratio_ranges), auc_l))
    show_data_auc_var = np.zeros((len(ratio_ranges), auc_l))
    
    file_name = "./judge_data/different_ratio/" + str(net_name) + "7base_different_edge_ratio_auc_data.pkl"
    if os.path.exists(file_name) is False:
        auc_data = [np.zeros((len(ratio_ranges), auc_l)) for j in range(times)]  # [{ratio_num, method_num}, , ...]  # auc_l就是基模型的个数
        with open(file_name, 'wb') as f:
            pickle.dump(auc_data, f)
    else:
        with open(file_name, 'rb') as f:
            auc_data = pickle.load(f)
    direct_get_result = False
    if direct_get_result is False:
        if not multi_ratio_process:
            for TRAIN_EDGE_RATIO in ratio_ranges:
                single_ratio_process(net_name, ratio_ranges, TRAIN_EDGE_RATIO, judge_methods, auc_data, file_name)
        else:
            lock = Manager().Lock()
            print("Pid:", os.getpid(), " Multiprocess model train")
            with get_context("spawn").Pool(processes=6) as model_ps:
                for TRAIN_EDGE_RATIO in ratio_ranges:
                    times = 20 if TRAIN_EDGE_RATIO < 0.1 else 10
                    for t in range(times):
                        base_train_edges, ensemble_train_edges, test_edges = get_train_test_data(net_name, t, TRAIN_EDGE_RATIO)
                        saveorload_path = "./model/DIFFRATIO" + "_" + net_name + "_time" + str(t) + "_ratio" + str(TRAIN_EDGE_RATIO) + "/"
                        model_file_name = saveorload_path + "Ensemble/weights.pkl"
                        if os.path.exists(model_file_name):
                            print(net_name, "ratio:", TRAIN_EDGE_RATIO, " time:", t, " Ensemble model has trained.")
                        else:
                            model = EnsembleJudge(net_name, base_train_edges, test_edges, judge_methods, ensemble_train_edges,
                                                  train_edge_ratio=TRAIN_EDGE_RATIO,
                                                  test_ep_ratio=1,
                                                  save=True, save_path=saveorload_path,
                                                  load=True, load_path=saveorload_path,
                                                  notrain=True)
                            for method in model.judge_methods:
                                base_model = model_ps.apply_async(func=model.get_model, args=(method, model.train_eps, model.train_news,
                                                                            model.save, model.save_path,
                                                                            model.load, model.load_path, ))
                    print("ratio:", TRAIN_EDGE_RATIO, " model train added.")
                model_ps.close()
                model_ps.join()
            print("Multiprocess ratio auc")
            with get_context("spawn").Pool(processes=6) as ps:
                for TRAIN_EDGE_RATIO in ratio_ranges:
                    ps.apply_async(single_ratio_process, args=(net_name, ratio_ranges, TRAIN_EDGE_RATIO, judge_methods,
                                                               auc_data, file_name, lock,))
                ps.close()
                ps.join()
                with open(file_name, 'rb') as f:
                    auc_data = pickle.load(f)
    else:
        pass
    for i in range(len(ratio_ranges)):
        for j in range(auc_l):
            aucs = [auc[i, j] for auc in auc_data if auc[i, j] != 0]
            show_data_auc_mean[i, j] = np.mean(aucs)
            show_data_auc_var[i, j] = np.sqrt(np.var(aucs))

    with open("./judge_data/different_ratio/" + str(net_name) + "mean_auc_data_for_train_edge_ratio.pkl", 'wb') as f:
        pickle.dump(show_data_auc_mean, f)
    with open("./judge_data/different_ratio/" + str(net_name) + "var_auc_data_for_train_edge_ratio.pkl", 'wb') as f:
        pickle.dump(show_data_auc_var, f)
    SHOW_METHODS = judge_methods + ['ensemble'] + EDGE_FEATURE
    print(net_name + ":")
    for method, method_i in zip(SHOW_METHODS, range(len(SHOW_METHODS))):
        print("---- " + method + ' :')
        print("Accuracy: ", dict(zip(ratio_ranges, show_data_auc_mean[:, method_i])))
        print("Std: ", dict(zip(ratio_ranges, show_data_auc_var[:, method_i])))
    # Excel
    sheet = excel_f.add_sheet(net_name, cell_overwrite_ok=True)
    for method, method_i in zip(SHOW_METHODS, range(len(SHOW_METHODS))):
        sheet.write_merge(2 * method_i + 1, 2 * method_i + 2, 0, 0, method)
        sheet.write(2 * method_i + 1, 1, "accu")
        sheet.write(2 * method_i + 2, 1, "std")
        for ratio, ratio_i in zip(ratio_ranges, range(len(ratio_ranges))):
            sheet.write(0, ratio_i + 2, ratio)
            sheet.write(2 * method_i + 1, ratio_i + 2, show_data_auc_mean[ratio_i, method_i])
            sheet.write(2 * method_i + 2, ratio_i + 2, show_data_auc_var[ratio_i, method_i])
    excel_f.save("./Result/different_ratio/" + net_name + "7base_different_ratio_accu.xls")


def show_edge_nums_with_time(_net_name):
    edge_days = get_all_edge_time(_net_name)
    edges_num_with_day = dict()
    for edge in edge_days.keys():
        if edge_days[edge] not in edges_num_with_day.keys():
            edges_num_with_day[edge_days[edge]] = 1
        else:
            edges_num_with_day[edge_days[edge]] += 1
    edge_num_with_time = dict()
    for i in sorted(edges_num_with_day):
        edge_num_with_time[i] = edges_num_with_day[i]
    return edge_num_with_time


def get_train_ep_ratio(net_name):
    ratio_ranges = np.round(list(np.arange(0.01, 0.1, 0.01)) + list(np.arange(0.1, 1, 0.1)), 3)
    edges_num = len(get_adj(net_name))
    ep_num = edges_num * (edges_num - 1) / 2
    edge_num_with_time = show_edge_nums_with_time(net_name)
    days = list(edge_num_with_time.keys())
    print(days)
    different_ep_num = 0
    for i in days:
        for j in days[days.index(i):]:
            if i != j:
                different_ep_num += edge_num_with_time[i] * edge_num_with_time[j]
    train_ep_ratios_mean = dict()
    train_ep_ratios_std = dict()
    for train_edge_ratio in ratio_ranges:
        times = 20 if train_edge_ratio < 0.1 else 10
        current_train_ep_ratios = []
        for t in range(times):
            print(net_name, train_edge_ratio, t, "Doing...")
            base_train_edges, ensemble_train_edges, test_edges = get_train_test_data(net_name, t, train_edge_ratio)
            train_ep_num = 0
            base_train_ep_num = 0
            ensemble_train_ep_num = 0
            for edge_pair, new in getEdgePair.get_single_eps_from_edges(base_train_edges, net_name):
                base_train_ep_num += 1
                train_ep_num += 1
            for edge_pair, new in getEdgePair.get_single_eps_from_edges(ensemble_train_edges, net_name):
                ensemble_train_ep_num += 1
                train_ep_num += 1
            for edge_pair, new in getEdgePair.get_single_eps_from_two_edges(base_train_edges, ensemble_train_edges, net_name):
                ensemble_train_ep_num += 1
                train_ep_num += 1
            current_train_ep_ratios.append(train_ep_num / ep_num)
        train_ep_ratios_mean[train_edge_ratio] = np.mean(current_train_ep_ratios)
        train_ep_ratios_std[train_edge_ratio] = np.std(current_train_ep_ratios)
        with open("./Result/different_ratio/" + net_name + "_edgeRatio2epRatio.txt", "a") as f:
            f.write(str(train_edge_ratio) + "\t" + str(train_ep_ratios_mean[train_edge_ratio]) + "\t" + str(train_ep_ratios_std[train_edge_ratio]) + "\n")
    with open("./Result/different_ratio/" + net_name + "_edgeRatio2epRatio.txt", "a") as f:
        f.write(str(1) + "\t" + str(different_ep_num/ep_num) + "\t" + str(0) + "\n")


if __name__ == '__main__':
    net_names = ["fruit_fly_mirrorTn%5"]
    for net_name in net_names:
        get_different_edge_ratio_auc(net_name, multi_ratio_process=True)

    # if you want to get the train ratio for edge paris, use this function
    # net_name = "chaos_new%"
    # for net_name in [net_name]:
    #     get_train_ep_ratio(net_name)
