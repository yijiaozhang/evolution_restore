import random
import math
import numpy as np
import time
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron, LogisticRegression
import getEdgePair
from __Configuration import *
import _PairStructureNNJudge_
import _BestSingleJudge_
import _SingleJudge_
import getEdgeEmbedding


class WeightSum:
    def __init__(self, train_x, train_y, save=False, save_path=None, load=False, load_path=None):
        """
        :rtype: object
        """
        self.train_x = train_x
        self.train_y = train_y
        self.train_num = np.shape(train_x)[0]
        self.weight_num = np.shape(train_x)[1]
        self.weights = np.zeros([1, self.weight_num])
        self.remain_weight = 1
        self.steps = 0.1
        self.iter = 0

        self.cur_best_auc = 0
        self.cur_best_weights = np.zeros([1, self.weight_num])
        if load and os.path.exists(load_path + "Ensemble/weights.pkl"):
            with open(load_path + "Ensemble/weights.pkl", "rb") as f:
                self.cur_best_weights = pickle.load(f)
            print(load_path + " Ensemble weight loaded")
        else:
            self.since = time.time()
            self.make_best_weights(0)
            if save:
                if os.path.exists(save_path + "Ensemble/") is False:
                    os.makedirs(save_path + "Ensemble/")
                with open(save_path + "Ensemble/weights.pkl", "wb") as f:
                    pickle.dump(self.cur_best_weights, f)

    def make_best_weights(self, weight_index=0):
        if weight_index < self.weight_num - 1:
            for w in np.arange(0, self.remain_weight, self.steps):
                self.weights[0, weight_index] = w
                self.remain_weight = self.remain_weight - w
                self.make_best_weights(weight_index+1)
                self.remain_weight = self.remain_weight + w
        elif weight_index == self.weight_num-1:
            self.weights[0, weight_index] = self.remain_weight
            tmp_auc = self.get_auc()
            if tmp_auc > self.cur_best_auc:
                self.cur_best_auc = tmp_auc
                self.cur_best_weights = np.copy(self.weights)
            self.iter += 1
            self.since = time.time()

    def get_auc(self):
        train_right_num = 0
        result = self.train_x.dot(self.weights.transpose())
        for i in range(self.train_num):
            if result[i, 0] > 0.5 and self.train_y[i, 0] == 1:
                train_right_num += 1
            elif result[i, 0] <= 0.5 and self.train_y[i, 0] == 0:
                train_right_num += 1
        auc = train_right_num / self.train_num
        return auc

    def predict(self, vec):
        judge = np.sum(vec * self.cur_best_weights)
        if judge > 0.5:
            return 1
        elif judge < 0.5:
            return 0
        else:
            a = random.random()
            if a < 0.5:
                return 1
            else:
                return 0


class EnsembleJudge:

    def __init__(self, net_name, train_edges, test_edges,
                 judge_methods=None,
                 ensemble_train_edges=None,
                 train_edge_ratio=0.1,
                 test_ep_ratio=1,
                 save=True,
                 save_path="./model/",
                 load=False,
                 load_path="./model/",
                 version='tensorflow',
                 notrain=False,
                 isTwoTrainSet=True):
        self.net_name = net_name
        self.train_edge_ratio = train_edge_ratio
        self.train_edges = train_edges
        self.test_edges = test_edges
        self.test_ep_ratio = test_ep_ratio
        self.ensemble_train_edges = ensemble_train_edges
        self.judge_methods = judge_methods
        self.version = version
        self.isTwoTrainSet = isTwoTrainSet
        self.model_num = len(self.judge_methods) if self.judge_methods is not None else 0
        self.train_eps = []
        self.train_news = []
        self.ensemble_train_eps = []
        self.ensemble_train_news = []
        # Ensemble get
        self.ensemble_clf = None
        self.base_models = []
        # save or load
        self.save = save
        self.save_path = save_path
        self.load = load
        self.load_path = load_path
        if self.train_edges is not None and self.test_edges is not None and self.ensemble_train_edges is not None:
            self.get_train_eps()
        self.ensemble_train_aucs = None
        if not notrain:
            self.ensemble_train()

    def get_train_eps(self):
        if self.isTwoTrainSet:
            train_ep_num = 0
            train_label1_num = 0
            for edge_pair, new in getEdgePair.get_single_eps_from_edges(self.train_edges, self.net_name):
                self.train_eps.append(edge_pair)
                self.train_news.append(new)
                train_ep_num += 1
                if new == 1:
                    train_label1_num += 1
            if self.ensemble_train_edges is not None:
                ensemble_train_ep_num = 0
                ensembel_train_label1_num = 0
                for edge_pair, new in getEdgePair.get_single_eps_from_edges(self.ensemble_train_edges, self.net_name):
                    self.ensemble_train_eps.append(edge_pair)
                    self.ensemble_train_news.append(new)
                    ensemble_train_ep_num += 1
                    if new == 1:
                        ensembel_train_label1_num += 1
                for edge_pair, new in getEdgePair.get_single_eps_from_two_edges(self.train_edges, self.ensemble_train_edges, self.net_name):
                    self.ensemble_train_eps.append(edge_pair)
                    self.ensemble_train_news.append(new)
                    ensemble_train_ep_num += 1
                    if new == 1:
                        ensembel_train_label1_num += 1
        else:
            if self.ensemble_train_edges is not None:
                train_ep_num = 0
                train_label1_num = 0
                ensemble_train_ep_num = 0
                ensembel_train_label1_num = 0
                for edge_pair, new in getEdgePair.get_single_eps_from_edges(self.train_edges+self.ensemble_train_edges, self.net_name):
                    self.train_eps.append(edge_pair)
                    self.train_news.append(new)
                    train_ep_num += 1
                    if new == 1:
                        train_label1_num += 1
                    self.ensemble_train_eps.append(edge_pair)
                    self.ensemble_train_news.append(new)
                    ensemble_train_ep_num += 1
                    if new == 1:
                        ensembel_train_label1_num += 1
                for edge_pair, new in getEdgePair.get_single_eps_from_two_edges(self.train_edges,
                                                                                self.ensemble_train_edges,
                                                                                self.net_name):
                    self.train_eps.append(edge_pair)
                    self.train_news.append(new)
                    train_ep_num += 1
                    if new == 1:
                        train_label1_num += 1
                    self.ensemble_train_eps.append(edge_pair)
                    self.ensemble_train_news.append(new)
                    ensemble_train_ep_num += 1
                    if new == 1:
                        ensembel_train_label1_num += 1

    def ensemble_train(self):
        print(self.net_name + " Ensembleing...")
        model_num = len(self.judge_methods)
        ensemble_train_ep_num = len(self.ensemble_train_eps)
        ensemble_train_x = np.zeros((ensemble_train_ep_num, model_num))
        ensemble_train_y = np.array([self.ensemble_train_news]).transpose()
        base_model_ensemble_aucs = []
        for method in self.judge_methods:
            print(self.net_name + " " + method + " begin...")
            method_index = self.judge_methods.index(method)
            model = self.get_model(method, _train_eps=self.train_eps, _train_news=self.train_news,
                                   save=self.save, save_path=self.save_path,
                                   load=self.load, load_path=self.load_path)
            self.base_models.append(model)
            method_judge_right_num = 0
            for ep, i in zip(self.ensemble_train_eps, list(range(ensemble_train_ep_num))):
                ensemble_train_x[i, method_index] = model.get_ep_judge_prob(ep)
                if ensemble_train_x[i, method_index] > 0.5 and ensemble_train_y[i, 0] == 1 or ensemble_train_x[i, method_index] < 0.5 and ensemble_train_y[i, 0] == 0:
                    method_judge_right_num += 1
                if ensemble_train_x[i, method_index] == 0.5:
                    method_judge_right_num += 0.5
            base_model_ensemble_aucs.append(method_judge_right_num/ensemble_train_ep_num if ensemble_train_ep_num != 0 else 0)
        # Ensemble Stacking Model Train
        print("Ensemble weight sum ing...")
        self.ensemble_clf = WeightSum(ensemble_train_x, ensemble_train_y,
                                      save=self.save, save_path=self.save_path,
                                      load=self.load, load_path=self.load_path)
        print("Ensemble Done. param:", self.ensemble_clf.cur_best_weights)
        ensemble_judge_right_num = 0
        for ep, i in zip(self.ensemble_train_eps, list(range(ensemble_train_ep_num))):
            ensemble_judge = self.ensemble_clf.predict(ensemble_train_x[i, :])
            if ensemble_judge == ensemble_train_y[i, 0]:
                ensemble_judge_right_num += 1
        self.ensemble_train_aucs = dict(zip(self.judge_methods + ["ensemble"], base_model_ensemble_aucs +
                                            [ensemble_judge_right_num/ensemble_train_ep_num if ensemble_train_ep_num != 0 else 0]))
        print(self.net_name + " Ensemble Model train ep num:" + str(ensemble_train_ep_num) + ". Train auc:")
        print(self.ensemble_train_aucs)

    def get_test_auc_two_train_edges_ensemble_stacking(self):
        model_num = len(self.judge_methods)
        test_num = 0
        test_right_num = [0] * (model_num + 1)
        since = time.time()
        for edge_pair, new in getEdgePair.get_single_eps_from_edges(self.test_edges, self.net_name, self.test_ep_ratio):
            test_num += 1
            base_model_vec = []
            for i in range(model_num):
                tmp_judge = self.base_models[i].get_ep_judge_prob(edge_pair)
                base_model_vec.append(tmp_judge)
                if tmp_judge > 0.5 and new == 1 or tmp_judge < 0.5 and new == 0:
                    test_right_num[i] += 1
                if tmp_judge == 0.5:
                    test_right_num[i] += 0.5
            if self.ensemble_clf.predict(base_model_vec) == new:
                test_right_num[-1] += 1
            if test_num == 100:
                print(self.net_name, "Ensemble Model ", "test edge pair num:" + str(test_num),
                      "time spend:" + str(time.time() - since))
            if test_num % 1000000 == 0:
                print(self.net_name, "Ensemble Model ", "test edge pair num:" + str(test_num),
                      "time spend:" + str(time.time() - since))
                since = time.time()
        for edge_pair, new in getEdgePair.get_single_eps_from_two_edges(self.train_edges, self.test_edges,
                                                                        self.net_name, self.test_ep_ratio):
            test_num += 1
            base_model_vec = []
            for i in range(model_num):
                tmp_judge = self.base_models[i].get_ep_judge_prob(edge_pair)
                base_model_vec.append(tmp_judge)
                if tmp_judge > 0.5 and new == 1 or tmp_judge < 0.5 and new == 0:
                    test_right_num[i] += 1
                if tmp_judge == 0.5:
                    test_right_num[i] += 0.5
            if self.ensemble_clf.predict(base_model_vec) == new:
                test_right_num[-1] += 1
            if test_num % 1000000 == 0:
                print(self.net_name, "Ensemble Model ", "test edge pair num:" + str(test_num),
                      "time spend:" + str(time.time() - since))
                since = time.time()
        for edge_pair, new in getEdgePair.get_single_eps_from_two_edges(self.ensemble_train_edges, self.test_edges,
                                                                        self.net_name, self.test_ep_ratio):
            test_num += 1
            base_model_vec = []
            for i in range(model_num):
                tmp_judge = self.base_models[i].get_ep_judge_prob(edge_pair)
                base_model_vec.append(tmp_judge)
                if tmp_judge > 0.5 and new == 1 or tmp_judge < 0.5 and new == 0:
                    test_right_num[i] += 1
                if tmp_judge == 0.5:
                    test_right_num[i] += 0.5
            if self.ensemble_clf.predict(base_model_vec) == new:
                test_right_num[-1] += 1
            if test_num % 1000000 == 0:
                print(self.net_name, "Ensemble Model ", "test edge pair num:" + str(test_num),
                      "time spend:" + str(time.time() - since))
                since = time.time()
        test_auc = [test_right_num[i] / test_num for i in range(len(self.judge_methods) + 1)]
        print(self.net_name + " Ensemble Model test ep num:" + str(test_num) + ". Test auc:")
        print(dict(zip(self.judge_methods + ["ensemble"], test_auc)))
        return dict(zip(self.judge_methods + ["ensemble"], test_auc))

    def get_test_auc_by_edges(self, edges):
        model_num = len(self.judge_methods)
        test_num = 0
        test_right_num = [0] * (model_num + 1)
        since = time.time()
        for edge_pair, new in getEdgePair.get_single_eps_from_edges(edges, self.net_name, self.test_ep_ratio):
            test_num += 1
            base_model_vec = []
            for i in range(model_num):
                tmp_judge = self.base_models[i].get_ep_judge_prob(edge_pair)
                base_model_vec.append(tmp_judge)
                if tmp_judge > 0.5 and new == 1 or tmp_judge < 0.5 and new == 0:
                    test_right_num[i] += 1
                if tmp_judge == 0.5:
                    test_right_num[i] += 0.5
            if self.ensemble_clf.predict(base_model_vec) == new:
                test_right_num[-1] += 1
            if test_num == 100:
                print(self.net_name, "Ensemble Model ", "test edge pair num:" + str(test_num), "time spend:" + str(time.time() - since))
            if test_num % 1000000 == 0:
                print(self.net_name, "Ensemble Model ", "test edge pair num:" + str(test_num), "time spend:" + str(time.time() - since))
                since = time.time()
        test_auc = [test_right_num[i] / test_num for i in range(len(self.judge_methods) + 1)]
        print(self.net_name + " Ensemble Model test ep num:" + str(test_num) + ". Test auc:")
        print(dict(zip(self.judge_methods + ["ensemble"], test_auc)))
        return dict(zip(self.judge_methods + ["ensemble"], test_auc))

    def get_ep_judge(self, edge_pair, feature_dict_dict=None):
        base_model_vec = []
        for i in range(self.model_num):
            if self.judge_methods[i] == UNION_PAIR_NN:
                tmp_judge = self.base_models[i].get_ep_judge_prob(edge_pair, feature_dict_dict)
            else:
                tmp_judge = self.base_models[i].get_ep_judge_prob(edge_pair, feature_dict_dict)
            base_model_vec.append(tmp_judge)
        # print("base_model_vec:", base_model_vec)
        vecs = self.ensemble_clf.predict(base_model_vec)
        # print(edge_pair, vecs)
        return vecs
        # return self.ensemble_clf.predict(base_model_vec)

    def get_model(self, judge_method, _train_eps=None, _train_news=None,
                  save=False, save_path=None, load=False, load_path=None) -> object:
        print("Pid:", os.getpid(), load_path if load else save_path, judge_method, "model getting...")
        if judge_method == BEST_SINGLE:
            single_judge = _BestSingleJudge_.BestSingleJudge(self.net_name, _train_eps, _train_news,
                                                             train_edge_ratio=self.train_edge_ratio,
                                                             save=save, save_path=save_path,
                                                             load=load, load_path=load_path)
            return single_judge
        if judge_method == NODE2VEC_PAIR_NN:
            embedding = getEdgeEmbedding.Node2vecEmbedding(self.net_name)
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding,
                                                                train_edge_ratio=self.train_edge_ratio,
                                                                save=save, save_path=save_path,
                                                                load=load, load_path=load_path, version=self.version)
            return model
        if judge_method == UNION_PAIR_NN:
            embedding = getEdgeEmbedding.SimpleEmbedding(self.net_name)
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding,
                                                                train_edge_ratio=self.train_edge_ratio,
                                                                save=save, save_path=save_path,
                                                                load=load, load_path=load_path, version=self.version)
            return model
        if judge_method == LINE_PAIR_NN:
            embedding = getEdgeEmbedding.LineEmbedding(self.net_name)
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding,
                                                                train_edge_ratio=self.train_edge_ratio,
                                                                save=save, save_path=save_path,
                                                                load=load, load_path=load_path, version=self.version)
            return model
        if judge_method == STRUC2VEC_PAIR_NN:
            embedding = getEdgeEmbedding.Struct2vecEmbedding(self.net_name)
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding,
                                                                train_edge_ratio=self.train_edge_ratio,
                                                                save=save, save_path=save_path,
                                                                load=load, load_path=load_path, version=self.version)
            return model
        if judge_method == DEEPWALK_PAIR_NN:
            embedding = getEdgeEmbedding.DeepwalkEmbedding(self.net_name)
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding,
                                                                train_edge_ratio=self.train_edge_ratio,
                                                                save=save, save_path=save_path,
                                                                load=load, load_path=load_path, version=self.version)
            return model
        if judge_method == SDNE_PAIR_NN:
            embedding = getEdgeEmbedding.SdneEmbedding(self.net_name)
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding,
                                                                train_edge_ratio=self.train_edge_ratio,
                                                                save=save, save_path=save_path,
                                                                load=load, load_path=load_path, version=self.version)
            return model
        if judge_method == BN_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "bn")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding,
                                                                train_edge_ratio=self.train_edge_ratio,
                                                                save=save, save_path=save_path,
                                                                load=load, load_path=load_path, version=self.version)
            return model
        if judge_method == CN_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "cn")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding,
                                                                train_edge_ratio=self.train_edge_ratio,
                                                                save=save, save_path=save_path,
                                                                load=load, load_path=load_path, version=self.version)
            return model
        if judge_method == DEGREE_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "degree")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding,
                                                                train_edge_ratio=self.train_edge_ratio,
                                                                save=save, save_path=save_path,
                                                                load=load, load_path=load_path, version=self.version)
            return model
        if judge_method == STRENGTH_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "strength")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding,
                                                                train_edge_ratio=self.train_edge_ratio,
                                                                save=save, save_path=save_path,
                                                                load=load, load_path=load_path, version=self.version)
            return model
        if judge_method == CC_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "cc")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding,
                                                                train_edge_ratio=self.train_edge_ratio,
                                                                save=save, save_path=save_path,
                                                                load=load, load_path=load_path, version=self.version)
            return model
        if judge_method == RA_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "ra")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding,
                                                                train_edge_ratio=self.train_edge_ratio,
                                                                save=save, save_path=save_path,
                                                                load=load, load_path=load_path, version=self.version)
            return model
        if judge_method == AA_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "aa")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding,
                                                                train_edge_ratio=self.train_edge_ratio,
                                                                save=save, save_path=save_path,
                                                                load=load, load_path=load_path, version=self.version)
            return model
        if judge_method == PA_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "pa")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding,
                                                                train_edge_ratio=self.train_edge_ratio,
                                                                save=save, save_path=save_path,
                                                                load=load, load_path=load_path, version=self.version)
            return model
        if judge_method == LP_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "lp")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding,
                                                                train_edge_ratio=self.train_edge_ratio,
                                                                save=save, save_path=save_path,
                                                                load=load, load_path=load_path, version=self.version)
            return model
        if judge_method == KSHELL_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "k_shell")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding,
                                                                train_edge_ratio=self.train_edge_ratio,
                                                                save=save, save_path=save_path,
                                                                load=load, load_path=load_path, version=self.version)
            return model
        if judge_method == PR_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "pr")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding,
                                                                train_edge_ratio=self.train_edge_ratio,
                                                                save=save, save_path=save_path,
                                                                load=load, load_path=load_path, version=self.version)
            return model
        if judge_method == ENSEMBLE:
            return self


def main():
    # ensemble test
    global TRAIN_EDGE_RATIO
    TRAIN_EDGE_RATIO = 0.4
    net_names = protein_net_names_mirrorTn
    judge_methods = [BEST_SINGLE, NODE2VEC_PAIR_NN, UNION_PAIR_NN]
    times = 10
    show_data_auc_mean = np.zeros((len(net_names), len(judge_methods)+1))
    show_data_auc_var = np.zeros((len(net_names), len(judge_methods)+1))
    auc_data = [np.zeros((len(net_names), len(judge_methods) + 1)) for j in range(times)]
    file_name = ".//judge_data//ensemble//auc_data.pkl"
    if os.path.exists(file_name) is False:
        for t in range(times):
            print("Times", t, "Begin...")
            for net_name, i in zip(net_names, range(len(net_names))):
                base_train_edges, ensemble_train_edges, test_edges = getEdgePair.ensemble_get_train_test_edges(TRAIN_EDGE_RATIO, net_name)
                test_auc = EnsembleJudge(net_name, base_train_edges, test_edges, judge_methods, ensemble_train_edges).get_test_auc_two_train_edges_ensemble_stacking()
                auc_data[t][i, :] = list(test_auc.values())
        with open(file_name, 'wb') as f:
            pickle.dump(auc_data, f)
    else:
        with open(file_name, 'rb') as f:
            pickle.load(auc_data, f)

    for i in range(len(net_names)):
        for j in range(len(judge_methods)+1):
            aucs = [auc[i, j] for auc in auc_data]
            show_data_auc_mean[i, j] = np.mean(aucs)
            show_data_auc_var[i, j] = np.var(aucs)

    width = 0.2
    plt.figure(figsize=(15, 9))
    x = list(range(len(net_names)))
    x_ticks_pos = [x[i] + width * len(judge_methods) / 2 for i in range(len(net_names))]
    x_ticks = [net_names[i] for i in range(len(net_names))]
    plt.xticks(x_ticks_pos, x_ticks, fontsize=15)
    plt.yticks(fontsize=15)
    for i in range(len(judge_methods)+1):
        y = np.round(show_data_auc_mean[:, i], 3)
        yerr = show_data_auc_var[:, i]
        plt.bar(x, y, yerr=yerr, width=width, label=(judge_methods+['ensemble'])[i]+'_test')
        for a, b in zip(x, y):
            plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
        for j in range(len(x)):
            x[j] = x[j] + width
    plt.title("New old judge auc for networks and methods -- train_edge_ratio:" + str(TRAIN_EDGE_RATIO),
              fontsize=25, fontweight='bold')
    plt.xlabel("networks", fontsize=20, fontweight='bold')
    plt.ylabel("auc", fontsize=20, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig("./Result/protein/" + 'AVERAGE train_edge_ratio' + str(TRAIN_EDGE_RATIO) + "_ensemble_test_auc.png", dpi=600)
    # plt.savefig("./Result/social/" + 'train_edge_ratio' + str(TRAIN_EDGE_RATIO) + "_ensemble_test_auc.png", dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
