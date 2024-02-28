import numpy as np
import time
import getEdgePair
from __Configuration import *
import _PairStructureNNJudge_
import _BestSingleJudge_
import getEdgeEmbedding


class BaseModelJudge:

    def __init__(self, net_name, train_edges, test_edges, judge_methods=None, ensemble_train_edges=None, train_edge_ratio=0.1):
        self.net_name = net_name
        self.train_edge_ratio = train_edge_ratio
        self.train_edges = train_edges
        self.test_edges = test_edges
        self.test_ep_ratio = 1
        self.ensemble_train_edges = ensemble_train_edges
        self.judge_methods = judge_methods
        self.model_num = len(self.judge_methods) if self.judge_methods is not None else 0
        self.train_eps = []
        self.train_news = []
        self.ensemble_train_eps = []
        self.ensemble_train_news = []
        self.base_models = []
        self.get_train_eps()
        self.base_train()

    def get_train_eps(self):
        train_ep_num = 0
        train_label1_num = 0
        for edge_pair, new in getEdgePair.get_double_eps_from_edges(self.train_edges, self.net_name):
            self.train_eps.append(edge_pair)
            self.train_news.append(new)
            train_ep_num += 1
            if new == 1:
                train_label1_num += 1
        print(self.net_name, "ensemble model: (base) train edge pair num:" + str(train_ep_num),
              "label 1 sample num:" + str(train_label1_num))
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
            print(self.net_name, "ensemble model: ensemble train edge pair num:" + str(ensemble_train_ep_num),
                  "label 1 sample num:" + str(ensembel_train_label1_num))

    def base_train(self):
        """
        :return:
        """
        print(self.net_name + " Ensembleing...")
        model_num = len(self.judge_methods)
        ensemble_train_ep_num = len(self.ensemble_train_eps)
        ensemble_train_x = np.zeros((ensemble_train_ep_num, model_num))  # [ensemble_train_ep_num, model_num]
        ensemble_train_y = np.array([self.ensemble_train_news]).transpose()
        base_model_ensemble_aucs = []
        for method in self.judge_methods:
            print(self.net_name + " " + method + " begin...")
            method_index = self.judge_methods.index(method)
            model = self.get_model(method, _train_eps=self.train_eps, _train_news=self.train_news)
            self.base_models.append(model)
            method_judge_right_num = 0
            for ep, i in zip(self.ensemble_train_eps, list(range(ensemble_train_ep_num))):
                ensemble_train_x[i, method_index] = model.get_ep_judge_prob(ep)
                if ensemble_train_x[i, method_index] > 0.5 and ensemble_train_y[i, 0] == 1 or ensemble_train_x[i, method_index] < 0.5 and ensemble_train_y[i, 0] == 0:
                    method_judge_right_num += 1
                if ensemble_train_x[i, method_index] == 0.5:
                    method_judge_right_num += 0.5
            base_model_ensemble_aucs.append(method_judge_right_num / ensemble_train_ep_num)
        ensemble_train_aucs = dict(zip(self.judge_methods, base_model_ensemble_aucs))
        print(self.net_name + " ensemble train ep num:" + str(ensemble_train_ep_num) + ". base models auc(in ensemble train):")
        print(ensemble_train_aucs)

    def get_test_auc_two_train_edges_ensemble_stacking(self):
        model_num = len(self.judge_methods)
        test_num = 0
        test_right_num = [0] * model_num
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
            if test_num == 100:
                print(self.net_name, "Base Models ", "test edge pair num:" + str(test_num),
                      "time spend:" + str(time.time() - since))
            if test_num % 1000000 == 0:
                print(self.net_name, "Base Models ", "test edge pair num:" + str(test_num),
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
            if test_num % 1000000 == 0:
                print(self.net_name, "Base Models ", "test edge pair num:" + str(test_num),
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
            if test_num % 1000000 == 0:
                print(self.net_name, "Base Models ", "test edge pair num:" + str(test_num),
                      "time spend:" + str(time.time() - since))
                since = time.time()
        test_auc = [test_right_num[i] / test_num for i in range(len(self.judge_methods))]
        print(self.net_name + " Base Models test ep num:" + str(test_num) + ". Test auc:")
        print(dict(zip(self.judge_methods, test_auc)))
        return dict(zip(self.judge_methods, test_auc))

    def get_model(self, judge_method, _train_eps=None, _train_news=None):
        if judge_method == BEST_SINGLE:
            single_judge = _BestSingleJudge_.BestSingleJudge(self.net_name, _train_eps, _train_news, train_edge_ratio=self.train_edge_ratio)
            return single_judge
        if judge_method == NODE2VEC_PAIR_NN:
            embedding = getEdgeEmbedding.Node2vecEmbedding(self.net_name)
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding, train_edge_ratio=self.train_edge_ratio)
            return model
        if judge_method == UNION_PAIR_NN:
            embedding = getEdgeEmbedding.SimpleEmbedding(self.net_name)
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding, train_edge_ratio=self.train_edge_ratio)
            return model
        if judge_method == LINE_PAIR_NN:
            embedding = getEdgeEmbedding.LineEmbedding(self.net_name)
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding, train_edge_ratio=self.train_edge_ratio)
            return model
        if judge_method == STRUC2VEC_PAIR_NN:
            embedding = getEdgeEmbedding.Struct2vecEmbedding(self.net_name)
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding, train_edge_ratio=self.train_edge_ratio)
            return model
        if judge_method == DEEPWALK_PAIR_NN:
            embedding = getEdgeEmbedding.DeepwalkEmbedding(self.net_name)
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding, train_edge_ratio=self.train_edge_ratio)
            return model
        if judge_method == SDNE_PAIR_NN:
            embedding = getEdgeEmbedding.SdneEmbedding(self.net_name)
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding, train_edge_ratio=self.train_edge_ratio)
            return model
        if judge_method == BN_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "bn")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding, train_edge_ratio=self.train_edge_ratio)
            return model
        if judge_method == CN_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "cn")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding, train_edge_ratio=self.train_edge_ratio)
            return model
        if judge_method == DEGREE_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "degree")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding, train_edge_ratio=self.train_edge_ratio)
            return model
        if judge_method == STRENGTH_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "strength")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding, train_edge_ratio=self.train_edge_ratio)
            return model
        if judge_method == CC_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "cc")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding, train_edge_ratio=self.train_edge_ratio)
            return model
        if judge_method == RA_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "ra")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding, train_edge_ratio=self.train_edge_ratio)
            return model
        if judge_method == AA_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "aa")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding, train_edge_ratio=self.train_edge_ratio)
            return model
        if judge_method == PA_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "pa")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding, train_edge_ratio=self.train_edge_ratio)
            return model
        if judge_method == LP_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "lp")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding, train_edge_ratio=self.train_edge_ratio)
            return model
        if judge_method == KSHELL_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "k_shell")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding, train_edge_ratio=self.train_edge_ratio)
            return model
        if judge_method == PR_PAIR_NN:
            embedding = getEdgeEmbedding.SingleEmbedding(self.net_name, "pr")
            model = _PairStructureNNJudge_.PairStructureNNJudge(self.net_name, _train_eps, _train_news, embedding, train_edge_ratio=self.train_edge_ratio)
            return model
