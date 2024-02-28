import time
import os
import numpy as np
import pickle
import xlwt
import getEdgePair
from __Configuration import *
from _BaseModelJudge_ import BaseModelJudge


def get_all_accu():
    global TRAIN_EDGE_RATIO
    TRAIN_EDGE_RATIO = 0.4
    net_names = ["fruit_fly_mirrorTn%5"]
    judge_methods = [BN_PAIR_NN, CN_PAIR_NN, DEGREE_PAIR_NN, STRENGTH_PAIR_NN, CC_PAIR_NN, RA_PAIR_NN, AA_PAIR_NN, PA_PAIR_NN, LP_PAIR_NN, KSHELL_PAIR_NN, PR_PAIR_NN]
    times = 10
    auc_l = len(judge_methods)
    show_data_auc_mean = dict()
    show_data_auc_var = dict()
    all_auc_data = [dict() for j in range(times)]
    file_name = "./judge_data/accu/single_base_auc_data.pkl"
    excel_f = xlwt.Workbook()
    sheet = excel_f.add_sheet("accu", cell_overwrite_ok=True)
    if os.path.exists(file_name) is False:
        auc_data = {'auc': all_auc_data}
    else:
        with open(file_name, 'rb') as f:
            auc_data = pickle.load(f)
            all_auc_data = auc_data['auc']
    for net_name in net_names:
        if net_name in auc_data['auc'][0].keys():
            print(net_name, "had done!")
        else:
            for t in range(times):
                print(net_name, "Times", t, "Begin...")
                base_train_edges, ensemble_train_edges, test_edges = getEdgePair.ensemble_get_train_test_edges(TRAIN_EDGE_RATIO, net_name)
                test_auc = BaseModelJudge(net_name, base_train_edges, test_edges, judge_methods, ensemble_train_edges, train_edge_ratio=TRAIN_EDGE_RATIO).get_test_auc_two_train_edges_ensemble_stacking()
                all_auc_data[t][net_name] = list(test_auc.values())
            auc_data = {'auc': all_auc_data}
            with open(file_name, 'wb') as f:
                pickle.dump(auc_data, f)
        show_data_auc_mean[net_name] = [0] * auc_l
        show_data_auc_var[net_name] = [0] * auc_l
        for j in range(auc_l):
            aucs = [auc[net_name][j] for auc in auc_data['auc']]
            show_data_auc_mean[net_name][j] = np.mean(aucs)
            show_data_auc_var[net_name][j] = np.sqrt(np.var(aucs))
        with open("./judge_data/accu/single_base_mean_auc_data.pkl", 'wb') as f:
            pickle.dump(show_data_auc_mean, f)
        with open("./judge_data/accu/single_base_var_auc_data.pkl", 'wb') as f:
            pickle.dump(show_data_auc_var, f)
        print(net_name + ':')
        l = len(judge_methods)
        auc_with_var = []
        for j in range(l):
            auc_with_var.append(str(show_data_auc_mean[net_name][j]) + ' # ' + str(show_data_auc_var[net_name][j]))
        print(dict(zip(judge_methods, auc_with_var)))
        # Excel
        SHOW_METHODS = judge_methods
        net_name_i = net_names.index(net_name)
        sheet.write_merge(2 * net_name_i + 1, 2 * net_name_i + 2, 0, 0, net_name)
        sheet.write(2 * net_name_i + 1, 1, "accu")
        sheet.write(2 * net_name_i + 2, 1, "std")
        for method, method_i in zip(SHOW_METHODS, range(l)):
            sheet.write(0, method_i + 2, method)
            sheet.write(2 * net_name_i + 1, method_i + 2, show_data_auc_mean[net_name][method_i])
            sheet.write(2 * net_name_i + 2, method_i + 2, show_data_auc_var[net_name][method_i])
        excel_f.save("./Result/Accu/single_base_accu.xls")


if __name__ == '__main__':
    get_all_accu()
