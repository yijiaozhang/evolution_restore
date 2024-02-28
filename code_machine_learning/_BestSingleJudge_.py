import time
import os
import random
from getNet import get_mat
from getFeatureJudge import get_feature_dict, judge_by_feature
from __Configuration import *


class BestSingleJudge:
    def __init__(self, net_name, train_eps, train_news, train_edge_ratio,
                 save=False, save_path=None, load=False, load_path=None):
        self.net_name = net_name
        self.train_eps = train_eps
        self.train_news = train_news
        self.train_edge_ratio = train_edge_ratio
        self.feature_dict_dict = dict()
        net_mat = get_mat(self.net_name)
        for edge_feature in EDGE_FEATURE:
            self.feature_dict_dict[edge_feature] = get_feature_dict(net_mat, edge_feature, net_name)
        self.best_feature = None
        self.is_big_old_for_best_feature = True
        # save or load
        self.save = save
        self.save_path = save_path
        if load and os.path.exists(load_path + "BestSingle/best_feature.txt"):
            with open(load_path + "BestSingle/best_feature.txt", "r") as f:
                self.best_feature = f.readline().strip()
                self.is_big_old_for_best_feature = True if f.readline().strip() == 'True' else False
            print(self.net_name, "best feature:" + self.best_feature,
                  "is big old:" + str(self.is_big_old_for_best_feature), "loaded")
        else:
            self.get_best_feature()


    def get_model_name(self):
        return BEST_SINGLE

    def get_best_feature(self):
        train_aucs = dict()
        is_big_olds = dict()
        train_ep_num = 0
        for edge_feature in EDGE_FEATURE:
            feature_dict = self.feature_dict_dict[edge_feature]
            train_judge_right_num = 0
            train_ep_num = 0
            for edge_pair, new in zip(self.train_eps, self.train_news):
                train_ep_num += 1
                feature = judge_by_feature(edge_feature, feature_dict, edge_pair)
                if feature == new:
                    train_judge_right_num += 1
                elif feature == -1:
                    train_judge_right_num += 0.5
            train_aucs[edge_feature] = train_judge_right_num / train_ep_num
            is_big_old = True
            if train_aucs[edge_feature] < 0.5:
                is_big_old = False
                train_aucs[edge_feature] = 1 - train_aucs[edge_feature]
            is_big_olds[edge_feature] = is_big_old
        self.best_feature = max(train_aucs, key=train_aucs.get)
        self.is_big_old_for_best_feature = is_big_olds[self.best_feature]
        print(self.net_name, "best feature:" + self.best_feature, "is big old:" + str(self.is_big_old_for_best_feature),
              "best auc:" + str(train_aucs[self.best_feature]))
        if self.save:
            if os.path.exists(self.save_path + "BestSingle/") is False:
                os.makedirs(self.save_path + "BestSingle/")
            with open(self.save_path + "BestSingle/best_feature.txt", "w") as f:
                f.write(self.best_feature)
                f.write("\n")
                f.write(str(self.is_big_old_for_best_feature))

    def get_ep_judge(self, _edge_pair):
        """
        :param _edge_pair:
        :return:
        """

        best_feature_dict = self.feature_dict_dict[self.best_feature]
        ori_judge = judge_by_feature(self.best_feature, best_feature_dict, _edge_pair)
        if ori_judge == -1:
            return 1 if random.random() > 0.5 else 0
        elif self.is_big_old_for_best_feature:
            return ori_judge
        else:
            return 1 - ori_judge

    def get_ep_judge_prob(self, edge_pair, feature_dict_dict=None):
        """
        :param edge_pair:
        :return:
        """
        if feature_dict_dict is None:
            best_feature_dict = self.feature_dict_dict[self.best_feature]
        else:
            best_feature_dict = feature_dict_dict[self.best_feature]
        bf1, bf2 = best_feature_dict[edge_pair[0]], best_feature_dict[edge_pair[1]]
        if bf1 == bf2:
            return 0.5
        elif self.is_big_old_for_best_feature:
            return bf2 / (bf1 + bf2)
        else:
            return bf1 / (bf1 + bf2)
