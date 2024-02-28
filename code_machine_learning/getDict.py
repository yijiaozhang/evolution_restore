import pickle
from __Configuration import *


def get_dict(file_name):
    """
    txt(node1,node2):value
    :param file_name:
    :return: {(node1,node2):value, ...}
    """
    with open(file_name, 'r') as f:
        map_dict = dict()
        for line in f:
            line = line.strip().split(':')
            map_dict[tuple(eval(line[0]))] = float(line[1])
        return map_dict


def show_pickle_dict(file_name):
    with open(file_name, 'rb') as f:
        auc_net_method = pickle.load(f)
    print(auc_net_method)


def main():
    file_name = "D:\\Code\\NewOldEdgeJudgement\\real_node\\J_model\\net_data\\coauthor\\ori_adj\\author_id_map.pickle"
    file_name = "./judge_data/ensemble/auc_data.pkl"
    show_pickle_dict(file_name)


if __name__ == '__main__':
    main()
