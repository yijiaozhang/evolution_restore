import numpy as np
import time
import os
import pickle
import random
import getEdgePair
from getNet import get_mat
from getFeatureJudge import get_feature_dict, judge_by_feature
from __Configuration import *


class SingleJudge:
    def __init__(self, net_name, train_edges, test_edges, void_edges=None):
        self.net_name = net_name
        self.train_edges = train_edges
        self.void_edges = void_edges
        self.test_edges = test_edges
        net_mat = get_mat(self.net_name)
        self.feature_dict_dict = dict()
        for edge_feature in EDGE_FEATURE:
            self.feature_dict_dict[edge_feature] = get_feature_dict(net_mat, edge_feature, net_name)
        self.is_big_olds = dict()
        self.train()

    def train(self):
        for edge_feature in EDGE_FEATURE:
            feature_dict = self.feature_dict_dict[edge_feature]
            train_judge_right_num = 0
            train_ep_num = 0
            for edge_pair, new in getEdgePair.get_double_eps_from_edges(self.train_edges, self.net_name):
                train_ep_num += 1
                feature = judge_by_feature(edge_feature, feature_dict, edge_pair)
                if feature == new:
                    train_judge_right_num += 1
                elif feature == -1:
                    train_judge_right_num += 0.5
            train_auc = train_judge_right_num / train_ep_num
            is_big_old = True
            if train_auc < 0.5:
                is_big_old = False
            self.is_big_olds[edge_feature] = is_big_old

    def get_ep_judge(self, edge_feature, edge_pair):
        feature_dict = self.feature_dict_dict[edge_feature]
        is_big_old = self.is_big_olds[edge_feature]
        feature_judge = judge_by_feature(edge_feature, feature_dict, edge_pair)
        if feature_judge == -1:
            return 1 if random.random() > 0.5 else 0
        if is_big_old:
            return feature_judge
        else:
            return 1 - feature_judge

    def all_feature_judge(self):
        train_aucs = dict()
        is_big_olds = dict()
        test_aucs = dict()
        since = time.time()
        for edge_feature in EDGE_FEATURE:
            feature_dict = self.feature_dict_dict[edge_feature]
            train_judge_right_num = 0
            train_ep_num = 0
            for edge_pair, new in getEdgePair.get_double_eps_from_edges(self.train_edges, self.net_name):
                train_ep_num += 1
                feature = judge_by_feature(edge_feature, feature_dict, edge_pair)
                if feature == new:
                    train_judge_right_num += 1
                elif feature == -1:
                    train_judge_right_num += 0.5
            train_aucs[edge_feature] = train_judge_right_num / train_ep_num
            print(self.net_name, edge_feature, "train ep num:" + str(train_ep_num) + " train_auc:" + str(train_aucs[edge_feature]), "time:" + str(time.time()-since))
            since = time.time()
            is_big_old = True
            if train_aucs[edge_feature] < 0.5:
                is_big_old = False
                train_aucs[edge_feature] = 1 - train_aucs[edge_feature]
            is_big_olds[edge_feature] = is_big_old
            test_judge_right_num = 0
            test_ep_num = 0
            for edge_pair, new in getEdgePair.get_single_eps_from_edges(self.test_edges, self.net_name):
                test_ep_num += 1
                feature = judge_by_feature(edge_feature, feature_dict, edge_pair)
                if is_big_old and feature == new:
                    test_judge_right_num += 1
                elif is_big_old is False and feature == 1 - new:
                    test_judge_right_num += 1
                elif feature == -1:
                    test_judge_right_num += 0.5
                if test_ep_num % 1000000 == 0:
                    print(self.net_name, "test_ep_num:", str(test_ep_num), "time spend:" + str(time.time() - since))
                    since = time.time()
            for edge_pair, new in getEdgePair.get_single_eps_from_two_edges(self.train_edges, self.test_edges, self.net_name):
                test_ep_num += 1
                feature = judge_by_feature(edge_feature, feature_dict, edge_pair)
                if is_big_old and feature == new:
                    test_judge_right_num += 1
                elif is_big_old is False and feature == 1 - new:
                    test_judge_right_num += 1
                elif feature == -1:
                    test_judge_right_num += 0.5
                if test_ep_num % 1000000 == 0:
                    print(self.net_name, "test_ep_num:", str(test_ep_num), "time spend:" + str(time.time() - since))
                    since = time.time()
            if self.void_edges is not None:
                for edge_pair, new in getEdgePair.get_single_eps_from_two_edges(self.void_edges, self.test_edges, self.net_name):
                    test_ep_num += 1
                    feature = judge_by_feature(edge_feature, feature_dict, edge_pair)
                    if is_big_old and feature == new:
                        test_judge_right_num += 1
                    elif is_big_old is False and feature == 1 - new:
                        test_judge_right_num += 1
                    elif feature == -1:
                        test_judge_right_num += 0.5
                    if test_ep_num % 1000000 == 0:
                        print(self.net_name, "test_ep_num:", str(test_ep_num), "time spend:" + str(time.time() - since))
                        since = time.time()
            test_aucs[edge_feature] = test_judge_right_num / test_ep_num
            print(self.net_name, edge_feature, "test ep num:" + str(test_ep_num), "test_auc:" + str(test_aucs[edge_feature]))
            since = time.time()
        return train_aucs, is_big_olds, test_aucs




