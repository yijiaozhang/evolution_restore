import time
import os
import numpy as np
import pickle
import xlwt
from collections import Counter
import getEdgePair
import getEdgeEmbedding
from __Configuration import *
from _EnsembleJudge_ import EnsembleJudge
from _SingleJudge_ import SingleJudge


def get_train_test_data(net_name, times, train_edge_ratio):
    if os.path.exists("./judge_data/accu/data/") is False:
        os.makedirs("./judge_data/accu/data/")
    file_name = "./judge_data/accu/data/" + net_name + "_times" + str(times) + "_ratio" + str(train_edge_ratio) + ".pkl"
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            base_train_edges, ensemble_train_edges, test_edges = pickle.load(f)
    else:
        base_train_edges, ensemble_train_edges, test_edges = getEdgePair.ensemble_get_train_test_edges(train_edge_ratio, net_name)
        with open(file_name, 'wb') as f:
            pickle.dump((base_train_edges, ensemble_train_edges, test_edges), f)
    return base_train_edges, ensemble_train_edges, test_edges


def pre_process(net_name):
    embedding = getEdgeEmbedding.Node2vecEmbedding(net_name)
    embedding = getEdgeEmbedding.SimpleEmbedding(net_name)
    embedding = getEdgeEmbedding.LineEmbedding(net_name)
    embedding = getEdgeEmbedding.Struct2vecEmbedding(net_name)
    embedding = getEdgeEmbedding.DeepwalkEmbedding(net_name)
    embedding = getEdgeEmbedding.SdneEmbedding(net_name)
    pass


def single_time_process(net_name, t, TRAIN_EDGE_RATIO, isTwoTrainSet, base_num, judge_methods, testornot, lock, all_auc_data, single_is_big_olds_data, train_auc_data, file_name):
    try:
        print(net_name, "Times", t, "Begin...")
        with open("./log/" + net_name + "accu_log.txt", "a") as f:
            f.write(net_name + "Times" + str(t) + "Begin...\n")
        since = time.time()
        base_train_edges, ensemble_train_edges, test_edges = get_train_test_data(net_name, t, TRAIN_EDGE_RATIO)
        if isTwoTrainSet:
            saveorload_path = "./model/ACCUbase" + str(base_num) + "_" + net_name + "_time" + str(t) + "_ratio" + str(TRAIN_EDGE_RATIO) + "/"
        else:
            saveorload_path = "./model/ACCUbase" + str(base_num) + "_" + net_name + "_time" + str(t) + "_ratio" + str(TRAIN_EDGE_RATIO) + "_1trainset/"
        with open("./log/" + net_name + "accu_log.txt", "a") as f:
            f.write(net_name + "Times" + str(t) + "Model Training...\n")
        model = EnsembleJudge(net_name, base_train_edges, test_edges, judge_methods, ensemble_train_edges,
                              train_edge_ratio=TRAIN_EDGE_RATIO,
                              save=True, save_path=saveorload_path,
                              load=True, load_path=saveorload_path,
                              isTwoTrainSet=isTwoTrainSet)
        if testornot:
            with open("./log/" + net_name + "accu_log.txt", "a") as f:
                f.write(net_name + "Times" + str(t) + "Testing...\n")
            test_auc = model.get_test_auc_two_train_edges_ensemble_stacking()
            single_judge = SingleJudge(net_name, base_train_edges, test_edges, ensemble_train_edges)
            _, single_is_big_olds, single_test_auc = single_judge.all_feature_judge()
            if lock is None:
                all_auc_data[t][net_name] = list(test_auc.values()) + list(single_test_auc.values())
                single_is_big_olds_data[t][net_name] = list(single_is_big_olds.values())
                train_auc = model.ensemble_train_aucs
                train_auc_data[t][net_name] = list(train_auc.values())
                print("NET:", net_name, "Times:", t, "Train_edge_ratio:", TRAIN_EDGE_RATIO,
                      " End. Time consume is " + str(time.time() - since))
                with open("./log/" + str(net_name) + "diff_log.txt", 'w') as log_f:
                    log_f.write("NET:" + net_name + "Times:" + str(t) + "Train_edge_ratio:" + str(
                        TRAIN_EDGE_RATIO) + " End. Time consume is " + str(time.time() - since) + "\n")
                with open(file_name, 'wb') as f:
                    auc_data = {'auc': all_auc_data, 'single_is_big_old': single_is_big_olds_data, 'train_auc': train_auc_data}
                    with open(file_name, 'wb') as f:
                        pickle.dump(auc_data, f)
            else:
                lock.acquire()
                with open(file_name, 'rb') as f:
                    auc_data = pickle.load(f)
                    all_auc_data = auc_data['auc']
                    single_is_big_olds_data = auc_data['single_is_big_old']
                    if 'train_auc' in auc_data.keys():
                        train_auc_data = auc_data['train_auc']
                all_auc_data[t][net_name] = list(test_auc.values()) + list(single_test_auc.values())
                single_is_big_olds_data[t][net_name] = list(single_is_big_olds.values())
                train_auc = model.ensemble_train_aucs
                train_auc_data[t][net_name] = list(train_auc.values())
                print("NET:", net_name, "Times:", t, "Train_edge_ratio:", TRAIN_EDGE_RATIO,
                      " End. Time consume is " + str(time.time() - since))
                with open("./log/" + str(net_name) + "diff_log.txt", 'w') as log_f:
                    log_f.write("NET:" + net_name + "Times:" + str(t) + "Train_edge_ratio:" + str(TRAIN_EDGE_RATIO) + " End. Time consume is " + str(time.time() - since) + "\n")
                auc_data = {'auc': all_auc_data, 'single_is_big_old': single_is_big_olds_data, 'train_auc': train_auc_data}
                with open(file_name, 'wb') as f:
                    pickle.dump(auc_data, f)
                lock.release()
    except Exception as ex:
        print(ex)


def get_all_accu(base_num, net_names, times, force_calc=False, testornot=True, isTwoTrainSet=True)  -> object:
    """
    :param base_num
    :param net_names
    :param times
    :param force_calc
    :param testornot
    :param isTwoTrainSet
    :return:
    """
    if base_num == 3:
        judge_methods = [BEST_SINGLE, NODE2VEC_PAIR_NN, UNION_PAIR_NN]
        file_name = "./judge_data/accu/3base_auc_data(change hidden layer num).pkl"
        test_mean_file_name = "./judge_data/accu/3base_mean_auc_data.pkl"
        test_var_file_name = "./judge_data/accu/3base_var_auc_data.pkl"
        train_mean_file_name = "./judge_data/accu/3base_mean_train_auc_data.pkl"
        train_var_file_name = "./judge_data/accu/3base_var_train_auc_data.pkl"
        excel_file_name = "./Result/Accu/3base_all_accu(change hidden layer num).xls"
    elif base_num == 7:
        judge_methods = [BEST_SINGLE, NODE2VEC_PAIR_NN, UNION_PAIR_NN, LINE_PAIR_NN, STRUC2VEC_PAIR_NN, DEEPWALK_PAIR_NN, SDNE_PAIR_NN]
        if isTwoTrainSet:
            file_name = "./judge_data/accu/"+'_'.join(net_names)+"7base_auc_data.pkl"
            test_mean_file_name = "./judge_data/accu/"+'_'.join(net_names)+"7base_mean_auc_data.pkl"
            test_var_file_name = "./judge_data/accu/"+'_'.join(net_names)+"7base_var_auc_data.pkl"
            train_mean_file_name = "./judge_data/accu/"+'_'.join(net_names)+"7base_mean_train_auc_data.pkl"
            train_var_file_name = "./judge_data/accu/"+'_'.join(net_names)+"7base_var_train_auc_data.pkl"
            excel_file_name = "./Result/Accu/"+'_'.join(net_names)+"7base_all_accu.xls"
        else:
            file_name = "./judge_data/accu/"+'_'.join(net_names)+"7base_auc_data_1trainset.pkl"
            test_mean_file_name = "./judge_data/accu/"+'_'.join(net_names)+"7base_mean_auc_data_1trainset.pkl"
            test_var_file_name = "./judge_data/accu/"+'_'.join(net_names)+"7base_var_auc_data_1trainset.pkl"
            train_mean_file_name = "./judge_data/accu/"+'_'.join(net_names)+"7base_mean_train_auc_data_1trainset.pkl"
            train_var_file_name = "./judge_data/accu/"+'_'.join(net_names)+"7base_var_train_auc_data_1trainset.pkl"
            excel_file_name = "./Result/Accu/"+'_'.join(net_names)+"7base_all_accu_1trainset.xls"
    else:
        print("base_num wrong", base_num)
        return None
    # ensemble test
    # global TRAIN_EDGE_RATIO
    TRAIN_EDGE_RATIO = 0.4
    auc_l = len(judge_methods) + 1 + len(EDGE_FEATURE)
    show_data_auc_mean = dict()
    show_data_auc_var = dict()
    show_data_train_auc_mean = dict()
    show_data_train_auc_var = dict()
    most_single_is_big_old = dict()
    all_auc_data = [dict() for j in range(times)]
    single_is_big_olds_data = [dict() for j in range(times)]
    
    excel_f = xlwt.Workbook()
    sheet = excel_f.add_sheet("accu", cell_overwrite_ok=True)
    train_sheet = excel_f.add_sheet("train_accu", cell_overwrite_ok=True)
    if os.path.exists(file_name) is False:
        auc_data = {'auc': all_auc_data, 'single_is_big_old': single_is_big_olds_data}
    else:
        with open(file_name, 'rb') as f:
            auc_data = pickle.load(f)
            all_auc_data = auc_data['auc']
            single_is_big_olds_data = auc_data['single_is_big_old']
    train_auc_data = [dict() for j in range(times)]
    for net_name in net_names:
        if net_name in auc_data['auc'][0].keys() and net_name in train_auc_data[0].keys() and not force_calc:
            print(net_name, "had done!")
        else:
            for t in range(times):
                print(net_name, "Times", t, "Begin...")
                base_train_edges, ensemble_train_edges, test_edges = get_train_test_data(net_name, t, TRAIN_EDGE_RATIO)
                if isTwoTrainSet:
                    saveorload_path = "./model/ACCUbase" + str(base_num) + "_" + net_name + "_time" + str(t) + \
                                      "_ratio" + str(TRAIN_EDGE_RATIO) + "/"
                else:
                    saveorload_path = "./model/ACCUbase" + str(base_num) + "_" + net_name + "_time" + str(t) + \
                                      "_ratio" + str(TRAIN_EDGE_RATIO) + "_1trainset/"
                model = EnsembleJudge(net_name, base_train_edges, test_edges, judge_methods, ensemble_train_edges,
                                      train_edge_ratio=TRAIN_EDGE_RATIO,
                                      save=True, save_path=saveorload_path,
                                      load=False, load_path=saveorload_path,
                                      isTwoTrainSet=isTwoTrainSet)
                if testornot:
                    test_auc = model.get_test_auc_two_train_edges_ensemble_stacking()
                    single_judge = SingleJudge(net_name, base_train_edges, test_edges, ensemble_train_edges)
                    _, single_is_big_olds, single_test_auc = single_judge.all_feature_judge()
                    all_auc_data[t][net_name] = list(test_auc.values()) + list(single_test_auc.values())
                    single_is_big_olds_data[t][net_name] = list(single_is_big_olds.values())
                    train_auc = model.ensemble_train_aucs
                    train_auc_data[t][net_name] = list(train_auc.values())
                    auc_data = {'auc': all_auc_data, 'single_is_big_old': single_is_big_olds_data}
                    with open(file_name, 'wb') as f:
                        pickle.dump(auc_data, f)
        if testornot:
            show_data_auc_mean[net_name] = [0] * auc_l
            show_data_auc_var[net_name] = [0] * auc_l
            most_single_is_big_old[net_name] = [0] * len(EDGE_FEATURE)
            for j in range(auc_l):
                aucs = [auc[net_name][j] for auc in auc_data['auc']]
                show_data_auc_mean[net_name][j] = np.mean(aucs)
                show_data_auc_var[net_name][j] = np.sqrt(np.var(aucs))
            for k in range(len(EDGE_FEATURE)):
                single_is_big_oldss = [is_big_old[net_name][k] for is_big_old in auc_data['single_is_big_old']]
                most_single_is_big_old[net_name][k] = Counter(single_is_big_oldss).most_common(1)[0][0]
            with open(test_mean_file_name, 'wb') as f:
                pickle.dump(show_data_auc_mean, f)
            with open(test_var_file_name, 'wb') as f:
                pickle.dump(show_data_auc_var, f)
            show_data_train_auc_mean[net_name] = [0] * (len(judge_methods) + 1)
            show_data_train_auc_var[net_name] = [0] * (len(judge_methods) + 1)
            for j in range(len(judge_methods) + 1):
                aucs = [auc[net_name][j] for auc in train_auc_data]
                show_data_train_auc_mean[net_name][j] = np.mean(aucs)
                show_data_train_auc_var[net_name][j] = np.sqrt(np.var(aucs))
            with open(train_mean_file_name, 'wb') as f:
                pickle.dump(show_data_train_auc_mean, f)
            with open(train_var_file_name, 'wb') as f:
                pickle.dump(show_data_train_auc_var, f)
            show_edge_feature = []
            for edge_feature, j in zip(EDGE_FEATURE, list(range(len(EDGE_FEATURE)))):
                if most_single_is_big_old[net_name][j] == 1:
                    s = 'b_o'
                else:
                    s = 's_o'
                show_edge_feature.append(edge_feature + "\n" + s)
            print(net_name + ':')
            l = len(judge_methods + ['ensemble'] + show_edge_feature)
            auc_with_var = []
            for j in range(l):
                auc_with_var.append(str(show_data_auc_mean[net_name][j]) + ' # ' + str(show_data_auc_var[net_name][j]))
            print(dict(zip(judge_methods + ['ensemble'] + show_edge_feature, auc_with_var)))
            # Excel
            SHOW_METHODS = judge_methods + ['ensemble'] + show_edge_feature
            net_name_i = net_names.index(net_name)
            sheet.write_merge(2 * net_name_i + 1, 2 * net_name_i + 2, 0, 0, net_name)
            sheet.write(2 * net_name_i + 1, 1, "accu")
            sheet.write(2 * net_name_i + 2, 1, "std")
            for method, method_i in zip(SHOW_METHODS, range(l)):
                sheet.write(0, method_i + 2, method)
                sheet.write(2 * net_name_i + 1, method_i + 2, show_data_auc_mean[net_name][method_i])
                sheet.write(2 * net_name_i + 2, method_i + 2, show_data_auc_var[net_name][method_i])
            SHOW_METHODS = judge_methods + ['ensemble']
            l = len(SHOW_METHODS)

            train_sheet.write_merge(2 * net_name_i + 1, 2 * net_name_i + 2, 0, 0, net_name)
            train_sheet.write(2 * net_name_i + 1, 1, "accu")
            train_sheet.write(2 * net_name_i + 2, 1, "std")
            for method, method_i in zip(SHOW_METHODS, range(l)):
                train_sheet.write(0, method_i + 2, method)
                train_sheet.write(2 * net_name_i + 1, method_i + 2, show_data_train_auc_mean[net_name][method_i])
                train_sheet.write(2 * net_name_i + 2, method_i + 2, show_data_train_auc_var[net_name][method_i])
            excel_f.save(excel_file_name)


def get_all_k_accu(base_num,  exterior_t, net_names, force_calc=False, testornot=True, isTwoTrainSet=True):
    """
    for link predition
    define t from outer layer
    """
    if base_num == 3:
        judge_methods = [BEST_SINGLE, NODE2VEC_PAIR_NN, UNION_PAIR_NN]
        file_name = "./judge_data/accu/3base_auc_data(change hidden layer num).pkl"
        test_mean_file_name = "./judge_data/accu/3base_mean_auc_data.pkl"
        test_var_file_name = "./judge_data/accu/3base_var_auc_data.pkl"
        train_mean_file_name = "./judge_data/accu/3base_mean_train_auc_data.pkl"
        train_var_file_name = "./judge_data/accu/3base_var_train_auc_data.pkl"
        excel_file_name = "./Result/Accu/3base_all_accu(change hidden layer num).xls"
    elif base_num == 7:
        judge_methods = [BEST_SINGLE, NODE2VEC_PAIR_NN, UNION_PAIR_NN, LINE_PAIR_NN, STRUC2VEC_PAIR_NN,
                         DEEPWALK_PAIR_NN, SDNE_PAIR_NN]
        if isTwoTrainSet:
            file_name = "./judge_data/accu/" + '_'.join(net_names) + "7base_auc_data.pkl"
            test_mean_file_name = "./judge_data/accu/" + '_'.join(net_names) + "7base_mean_auc_data.pkl"
            test_var_file_name = "./judge_data/accu/" + '_'.join(net_names) + "7base_var_auc_data.pkl"
            train_mean_file_name = "./judge_data/accu/" + '_'.join(net_names) + "7base_mean_train_auc_data.pkl"
            train_var_file_name = "./judge_data/accu/" + '_'.join(net_names) + "7base_var_train_auc_data.pkl"
            excel_file_name = "./Result/Accu/" + '_'.join(net_names) + "7base_all_accu.xls"
        else:
            file_name = "./judge_data/accu/" + '_'.join(net_names) + "7base_auc_data_1trainset.pkl"
            test_mean_file_name = "./judge_data/accu/" + '_'.join(net_names) + "7base_mean_auc_data_1trainset.pkl"
            test_var_file_name = "./judge_data/accu/" + '_'.join(net_names) + "7base_var_auc_data_1trainset.pkl"
            train_mean_file_name = "./judge_data/accu/" + '_'.join(
                net_names) + "7base_mean_train_auc_data_1trainset.pkl"
            train_var_file_name = "./judge_data/accu/" + '_'.join(net_names) + "7base_var_train_auc_data_1trainset.pkl"
            excel_file_name = "./Result/Accu/" + '_'.join(net_names) + "7base_all_accu_1trainset.xls"
    else:
        print("base_num wrong", base_num)
        return None

    times = 1
    TRAIN_EDGE_RATIO = 0.4
    auc_l = len(judge_methods) + 1 + len(EDGE_FEATURE)
    show_data_auc_mean = dict()
    show_data_auc_var = dict()
    show_data_train_auc_mean = dict()
    show_data_train_auc_var = dict()
    most_single_is_big_old = dict()
    all_auc_data = [dict() for j in range(times)]
    single_is_big_olds_data = [dict() for j in range(times)]

    excel_f = xlwt.Workbook()
    sheet = excel_f.add_sheet("accu", cell_overwrite_ok=True)
    train_sheet = excel_f.add_sheet("train_accu", cell_overwrite_ok=True)
    if os.path.exists(file_name) is False:
        auc_data = {'auc': all_auc_data, 'single_is_big_old': single_is_big_olds_data}
    else:
        with open(file_name, 'rb') as f:
            auc_data = pickle.load(f)
            all_auc_data = auc_data['auc']
            single_is_big_olds_data = auc_data['single_is_big_old']
    train_auc_data = [dict() for j in range(times)]
    for net_name in net_names:
        if net_name in auc_data['auc'][0].keys() and net_name in train_auc_data[0].keys() and not force_calc:
            print(net_name, "had done!")
        else:
            for t in range(times):
                print(net_name, "Times", exterior_t, "Begin...")
                base_train_edges, ensemble_train_edges, test_edges = get_train_test_data(net_name, exterior_t, TRAIN_EDGE_RATIO)
                if isTwoTrainSet:
                    saveorload_path = "./model/ACCUbase" + str(base_num) + "_" + net_name + "_time" + str(exterior_t) + \
                                      "_ratio" + str(TRAIN_EDGE_RATIO) + "/"
                else:
                    saveorload_path = "./model/ACCUbase" + str(base_num) + "_" + net_name + "_time" + str(exterior_t) + \
                                      "_ratio" + str(TRAIN_EDGE_RATIO) + "_1trainset/"
                model = EnsembleJudge(net_name, base_train_edges, test_edges, judge_methods, ensemble_train_edges,
                                      train_edge_ratio=TRAIN_EDGE_RATIO,
                                      save=True, save_path=saveorload_path,
                                      load=False, load_path=saveorload_path,
                                      isTwoTrainSet=isTwoTrainSet)
                if testornot:
                    test_auc = model.get_test_auc_two_train_edges_ensemble_stacking()
                    single_judge = SingleJudge(net_name, base_train_edges, test_edges, ensemble_train_edges)
                    _, single_is_big_olds, single_test_auc = single_judge.all_feature_judge()
                    all_auc_data[t][net_name] = list(test_auc.values()) + list(single_test_auc.values())
                    single_is_big_olds_data[t][net_name] = list(single_is_big_olds.values())
                    train_auc = model.ensemble_train_aucs
                    train_auc_data[t][net_name] = list(train_auc.values())
                    auc_data = {'auc': all_auc_data, 'single_is_big_old': single_is_big_olds_data}
                    with open(file_name, 'wb') as f:
                        pickle.dump(auc_data, f)
        if testornot:
            show_data_auc_mean[net_name] = [0] * auc_l
            show_data_auc_var[net_name] = [0] * auc_l
            most_single_is_big_old[net_name] = [0] * len(EDGE_FEATURE)
            for j in range(auc_l):
                aucs = [auc[net_name][j] for auc in auc_data['auc']]
                show_data_auc_mean[net_name][j] = np.mean(aucs)
                show_data_auc_var[net_name][j] = np.sqrt(np.var(aucs))
            for k in range(len(EDGE_FEATURE)):
                single_is_big_oldss = [is_big_old[net_name][k] for is_big_old in auc_data['single_is_big_old']]
                most_single_is_big_old[net_name][k] = Counter(single_is_big_oldss).most_common(1)[0][0]
            with open(test_mean_file_name, 'wb') as f:
                pickle.dump(show_data_auc_mean, f)
            with open(test_var_file_name, 'wb') as f:
                pickle.dump(show_data_auc_var, f)
            show_data_train_auc_mean[net_name] = [0] * (len(judge_methods) + 1)
            show_data_train_auc_var[net_name] = [0] * (len(judge_methods) + 1)
            for j in range(len(judge_methods) + 1):
                aucs = [auc[net_name][j] for auc in train_auc_data]
                show_data_train_auc_mean[net_name][j] = np.mean(aucs)
                show_data_train_auc_var[net_name][j] = np.sqrt(np.var(aucs))
            with open(train_mean_file_name, 'wb') as f:
                pickle.dump(show_data_train_auc_mean, f)
            with open(train_var_file_name, 'wb') as f:
                pickle.dump(show_data_train_auc_var, f)
            show_edge_feature = []
            for edge_feature, j in zip(EDGE_FEATURE, list(range(len(EDGE_FEATURE)))):
                if most_single_is_big_old[net_name][j] == 1:
                    s = 'b_o'
                else:
                    s = 's_o'
                show_edge_feature.append(edge_feature + "\n" + s)
            print(net_name + ':')
            l = len(judge_methods + ['ensemble'] + show_edge_feature)
            auc_with_var = []
            for j in range(l):
                auc_with_var.append(str(show_data_auc_mean[net_name][j]) + ' # ' + str(show_data_auc_var[net_name][j]))
            print(dict(zip(judge_methods + ['ensemble'] + show_edge_feature, auc_with_var)))
            SHOW_METHODS = judge_methods + ['ensemble'] + show_edge_feature
            net_name_i = net_names.index(net_name)
            sheet.write_merge(2 * net_name_i + 1, 2 * net_name_i + 2, 0, 0, net_name)
            sheet.write(2 * net_name_i + 1, 1, "accu")
            sheet.write(2 * net_name_i + 2, 1, "std")
            for method, method_i in zip(SHOW_METHODS, range(l)):
                sheet.write(0, method_i + 2, method)
                sheet.write(2 * net_name_i + 1, method_i + 2, show_data_auc_mean[net_name][method_i])
                sheet.write(2 * net_name_i + 2, method_i + 2, show_data_auc_var[net_name][method_i])
            SHOW_METHODS = judge_methods + ['ensemble']
            l = len(SHOW_METHODS)
            train_sheet.write_merge(2 * net_name_i + 1, 2 * net_name_i + 2, 0, 0, net_name)
            train_sheet.write(2 * net_name_i + 1, 1, "accu")
            train_sheet.write(2 * net_name_i + 2, 1, "std")
            for method, method_i in zip(SHOW_METHODS, range(l)):
                train_sheet.write(0, method_i + 2, method)
                train_sheet.write(2 * net_name_i + 1, method_i + 2, show_data_train_auc_mean[net_name][method_i])
                train_sheet.write(2 * net_name_i + 2, method_i + 2, show_data_train_auc_var[net_name][method_i])
            excel_f.save(excel_file_name)


if __name__ == '__main__':
    get_all_accu(7, net_names=["fruit_fly_mirrorTn%5"], times=10, force_calc=True, testornot=True, isTwoTrainSet=True)
