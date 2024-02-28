import time
import pickle
import os
import gc
import random
import numpy as np
import matplotlib.pyplot as plt
import getEdgePair
import getEdgeEmbedding
import tensorflow as tf
import xlwt
from multiprocessing import Manager, Process
from getNet import get_mat, divide_name
from keras.utils import np_utils
from __Configuration import *
from getFeatureJudge import get_feature_dict, judge_by_feature
from _EnsembleJudge_ import WeightSum, EnsembleJudge


class CrossNetsJudge:
    def __init__(self, train_nets, test_nets, train_edge_ratio, times, judge_method, merge_data_or_model="data", test_ep_ratio=1):
        """
        :param train_nets
        :param test_nets
        :param train_edge_ratio
        :param times
        :param judge_method
        :param merge_data_or_model
        :param test_ep_ratio
        """

        self.train_nets = train_nets
        self.test_nets = test_nets
        self.train_edge_ratio = train_edge_ratio
        self.judge_method = judge_method
        if test_nets[0] not in train_nets:
            self.test_edge_ratio = 1
        self.merge_data_or_model = merge_data_or_model
        self.test_ep_ratio = test_ep_ratio
        # path
        str_train_nets = ",".join(train_nets)
        self.data_path = "./judge_data/cross_data/data/"+str_train_nets+"_times"+str(times)+ "_ratio"+str(train_edge_ratio)+".pkl"
        self.model_path = "./model/CrossJudgebase"+str(7)+"_"+str_train_nets+"_time"+str(times)+"_ratio"+str(train_edge_ratio)+self.judge_method+"/"
        key = str(train_nets) + str(train_edge_ratio) + merge_data_or_model + str(times)
        # Ensemble
        self.base_method_num = 7
        self.best_feature = None
        self.node2vec = None
        self.collection = None
        self.line = None
        self.struc2vec = None
        self.deepwalk = None
        self.sdne = None
        self.ensemble_clf = None
        self.models = []
        self.clf = None
        self.edges_list = []
        print(key, "clf is construct..")
        if merge_data_or_model == "data":
            self.cross_nets_ensemble_bydata()
        elif merge_data_or_model == "model":
            self.model_num = len(self.train_nets)
            self.cross_nets_ensemble_bymodel()
        else:
            print("param wrong!!!!!")

    def cross_nets_ensemble_bydata(self):
        if os.path.exists(self.data_path):
            with open(self.data_path, "rb") as f:
                edges_list = pickle.load(f)
            for edges in edges_list:
                print(edges.net, "cross nets ensemble model: ", "(base) train edge pair num:" + str(len(edges.edge_pair)))
                print(edges.net, "cross nets ensemble model: ", "(ensemble) train edge pair num:" + str(len(edges.ensemble_edge_pair)))
        else:
            edges_list = [self.Edges(net, self.train_edge_ratio, ensemble=True) for net in self.train_nets]
            if os.path.exists("./judge_data/cross_data/data/") is False:
                os.makedirs("./judge_data/cross_data/data/")
            with open(self.data_path, "wb") as f:
                pickle.dump(edges_list, f)
        self.edges_list = edges_list
        if os.path.exists(self.model_path) is False:
            os.makedirs(self.model_path)
        if self.judge_method == BEST_SINGLE or self.judge_method == ENSEMBLE:
            self.best_feature = self.BestSingleJudge(edges_list, save=True, save_path=self.model_path , load=True, load_path=self.model_path)
        if self.judge_method == NODE2VEC_PAIR_NN or self.judge_method == ENSEMBLE:
            self.node2vec = self.PairNNJudge(edges_list, NODE2VEC, save=True, save_path=self.model_path, load=True, load_path=self.model_path)
        if self.judge_method == UNION_PAIR_NN or self.judge_method == ENSEMBLE:
            self.collection = self.PairNNJudge(edges_list, COLLECTION, save=True, save_path=self.model_path, load=True, load_path=self.model_path)
        if self.judge_method == LINE_PAIR_NN or self.judge_method == ENSEMBLE:
            self.line = self.PairNNJudge(edges_list, LINE_EMBEDDING, save=True, save_path=self.model_path , load=True, load_path=self.model_path)
        if self.judge_method == STRUC2VEC_PAIR_NN or self.judge_method == ENSEMBLE:
            self.struc2vec = self.PairNNJudge(edges_list, STRUCT2VEC_EMBEDDING, save=True, save_path=self.model_path , load=True, load_path=self.model_path)
        if self.judge_method == DEEPWALK_PAIR_NN or self.judge_method == ENSEMBLE:
            self.deepwalk = self.PairNNJudge(edges_list, DEEPWALK_EMBEDDING, save=True, save_path=self.model_path, load=True, load_path=self.model_path)
        if self.judge_method == SDNE_PAIR_NN or self.judge_method == ENSEMBLE:
            self.sdne = self.PairNNJudge(edges_list, SDNE_EMBEDDING, save=True, save_path=self.model_path , load=True, load_path=self.model_path)
        if self.judge_method == ENSEMBLE:
            ensemble_train_ep_num = 0
            for edges in edges_list:
                ensemble_train_ep_num += len(edges.ensemble_edge_pair)
            ensemble_train_x = np.zeros((ensemble_train_ep_num, self.base_method_num))
            ensemble_train_y = np.zeros((ensemble_train_ep_num, 1))
            models = [self.best_feature, self.node2vec, self.collection, self.line, self.struc2vec, self.deepwalk, self.sdne]
            since = time.time()
            print("Construct ensemble train set...")
            for model in models:
                model_index = models.index(model)
                ep_i = 0
                for edges in edges_list:
                    if model.embedding_name == BEST_SINGLE:
                        embedding = edges.feature_dict_dict[model.best_feature]
                    elif model.embedding_name == NODE2VEC:
                        embedding = edges.node2vec_embedding
                    elif model.embedding_name == COLLECTION:
                        embedding = edges.simple_embedding
                    elif model.embedding_name == LINE_EMBEDDING:
                        embedding = edges.line_embedding
                    elif model.embedding_name == STRUCT2VEC_EMBEDDING:
                        embedding = edges.struct2vec_embedding
                    elif model.embedding_name == DEEPWALK_EMBEDDING:
                        embedding = edges.deepwalk_embedding
                    elif model.embedding_name == SDNE_EMBEDDING:
                        embedding = edges.sdne_embedding
                    else:
                        embedding = None
                    ensemble_train_y[ep_i:ep_i+len(edges.ensemble_edge_pair), :] = np.array([list(edges.ensemble_edge_pair.values())]).transpose()
                    for ep in edges.ensemble_edge_pair.keys():
                        ensemble_train_x[ep_i, model_index] = model.get_ep_judge_prob(ep, embedding)
                        # ensemble_train_y[ep_i, 0] = edges.ensemble_edge_pair[ep]
                        ep_i += 1
                # print(model_index, ep_i, "Time consume:", time.time() - since)
                since = time.time()
            print("Ensemble weight sum ing...")
            self.ensemble_clf = WeightSum(ensemble_train_x, ensemble_train_y, save=True, save_path=self.model_path
                                          , load=True, load_path=self.model_path)
            print("Ensemble Done. param:", self.ensemble_clf.cur_best_weights, "best auc:", self.ensemble_clf.cur_best_auc)
        if self.judge_method == BEST_SINGLE:
            self.clf = self.best_feature
        if self.judge_method == NODE2VEC_PAIR_NN:
            self.clf = self.node2vec
        if self.judge_method == UNION_PAIR_NN:
            self.clf = self.collection
        if self.judge_method == LINE_PAIR_NN:
            self.clf = self.line
        if self.judge_method == STRUC2VEC_PAIR_NN:
            self.clf = self.struc2vec
        if self.judge_method == DEEPWALK_PAIR_NN:
            self.clf = self.deepwalk
        if self.judge_method == SDNE_PAIR_NN:
            self.clf = self.sdne
        if self.judge_method == ENSEMBLE:
            self.clf = self.ensemble_clf

    def cross_nets_ensemble_bymodel(self):
        ensemble_edges_list = []
        for net in self.train_nets:
            one_net_model = dict()
            edges = self.Edges(net, self.train_edge_ratio, ensemble=True)
            ensemble_edges_list.append(edges)
            edges_list = [edges]
            best_feature = self.BestSingleJudge(edges_list)
            node2vec = self.PairNNJudge(edges_list, NODE2VEC)
            collection = self.PairNNJudge(edges_list, COLLECTION)
            ensemble_train_ep_num = 0
            for edges in edges_list:
                ensemble_train_ep_num += len(edges.ensemble_edge_pair)
            ensemble_train_x = np.zeros((ensemble_train_ep_num, self.base_method_num))
            ensemble_train_y = np.zeros((ensemble_train_ep_num, 1))
            ep_i = 0
            for edges in edges_list:
                for ep in edges.ensemble_edge_pair:
                    ensemble_train_x[ep_i, :] = [best_feature.get_ep_judge_prob(ep, edges), node2vec.get_ep_judge_prob(ep, edges), collection.get_ep_judge_prob(ep, edges)]
                    ensemble_train_y[ep_i, 0] = edges.ensemble_edge_pair[ep]
                    ep_i += 1
            one_net_ensemble_clf = WeightSum(ensemble_train_x, ensemble_train_y)
            one_net_model[BEST_SINGLE] = best_feature
            one_net_model[NODE2VEC_PAIR_NN] = node2vec
            one_net_model[UNION_PAIR_NN] = collection
            one_net_model[ENSEMBLE] = one_net_ensemble_clf
            self.models.append(one_net_model)
            print(net, "ensemble model get! and append to models")
        self.edges_list = ensemble_edges_list
        ensemble_train_ep_num = 0
        for edges in ensemble_edges_list:
            ensemble_train_ep_num += len(edges.ensemble_edge_pair)
        ensemble_train_x = np.zeros((ensemble_train_ep_num, self.model_num))
        ensemble_train_y = np.zeros((ensemble_train_ep_num, 1))
        ep_i = 0
        for edges in ensemble_edges_list:
            for ep in edges.ensemble_edge_pair:
                ensemble_judges = []
                for model in self.models:
                    base_vec = [model[BEST_SINGLE].get_ep_judge_prob(ep, edges), model[NODE2VEC_PAIR_NN].get_ep_judge_prob(ep, edges), model[UNION_PAIR_NN].get_ep_judge_prob(ep, edges)]
                    one_net_ensemble_judge = model[ENSEMBLE].predict(base_vec)
                    ensemble_judges.append(one_net_ensemble_judge)
                ensemble_train_x[ep_i, :] = ensemble_judges
                ensemble_train_y[ep_i, 0] = edges.ensemble_edge_pair[ep]
                ep_i += 1
        self.ensemble_clf = WeightSum(ensemble_train_x, ensemble_train_y)
        print("Ensemble Done. param:", self.ensemble_clf.cur_best_weights, "best auc:", self.ensemble_clf.cur_best_auc)

    class Edges:
        def __init__(self, net, edge_ratio, ensemble=False):
            self.net = net
            self.edge_ratio = edge_ratio
            if ensemble:
                self.edge, self.ensemble_edge, self.test_edge = getEdgePair.ensemble_get_train_test_edges(self.edge_ratio, self.net)
                self.edge_pair, self.ensemble_edge_pair = self.get_edge_pair(self.edge, self.ensemble_edge)
            else:
                self.test_edge, _ = getEdgePair.get_train_test_edges(self.edge_ratio, self.net)
            # train embedding
            self.node2vec_embedding = getEdgeEmbedding.Node2vecEmbedding(self.net)
            self.simple_embedding = getEdgeEmbedding.SimpleEmbedding(self.net)
            self.line_embedding = getEdgeEmbedding.LineEmbedding(self.net)
            self.struct2vec_embedding = getEdgeEmbedding.Struct2vecEmbedding(self.net)
            self.deepwalk_embedding = getEdgeEmbedding.DeepwalkEmbedding(self.net)
            self.sdne_embedding = getEdgeEmbedding.SdneEmbedding(self.net)
            self.feature_dict_dict = dict()
            net_mat = get_mat(self.net)
            for edge_feature in EDGE_FEATURE:
                self.feature_dict_dict[edge_feature] = get_feature_dict(net_mat, edge_feature, self.net)

        def get_edge_pair(self, edge, ensemble_edge):
            train_ep_num = 0
            train_label1_num = 0
            train_eps = dict()
            ensemble_train_eps = dict()
            for edge_pair, new in getEdgePair.get_single_eps_from_edges(edge, self.net):
                train_eps[edge_pair] = new
                train_ep_num += 1
                if new == 1:
                    train_label1_num += 1
            print(self.net, "cross nets ensemble model: ", "(base) train edge pair num:" + str(train_ep_num),
                  "label 1 sample num:" + str(train_label1_num))
            if self.ensemble_edge is not None:
                ensemble_train_ep_num = 0
                ensembel_train_label1_num = 0
                for edge_pair, new in getEdgePair.get_single_eps_from_edges(ensemble_edge, self.net):
                    ensemble_train_eps[edge_pair] = new
                    ensemble_train_ep_num += 1
                    if new == 1:
                        ensembel_train_label1_num += 1
                for edge_pair, new in getEdgePair.get_single_eps_from_two_edges(edge, ensemble_edge, self.net):
                    ensemble_train_eps[edge_pair] = new
                    ensemble_train_ep_num += 1
                    if new == 1:
                        ensembel_train_label1_num += 1
                print(self.net, "cross nets ensemble model: ", "ensemble train edge pair num:" + str(ensemble_train_ep_num),
                      "label 1 sample num:" + str(ensembel_train_label1_num))
            return train_eps, ensemble_train_eps

    class BestSingleJudge:
        def __init__(self, edges_list, save=False, save_path=None, load=False, load_path=None):
            """
            :param edges_list: list of Edges object
            """
            self.edges_list = edges_list
            self.embedding_name = BEST_SINGLE
            self.best_feature = None
            self.is_big_old_for_best_feature = None
            if load and os.path.exists(load_path + "BestSingle/best_feature.txt"):
                with open(load_path + "BestSingle/best_feature.txt", "r") as f:
                    self.best_feature = f.readline().strip()
                    self.is_big_old_for_best_feature = bool(f.readline().strip())
                print(self.edges_list[0].net, "best feature:" + self.best_feature,
                      "is big old:" + str(self.is_big_old_for_best_feature), "loaded")
            else:
                self.get_best_feature()
                if save:
                    if os.path.exists(save_path + "BestSingle/") is False:
                        os.makedirs(save_path + "BestSingle/")
                    with open(save_path + "BestSingle/best_feature.txt", "w") as f:
                        f.write(self.best_feature)
                        f.write("\n")
                        f.write(str(self.is_big_old_for_best_feature))

        def get_best_feature(self):
            train_aucs = dict()
            is_big_olds = dict()
            since = time.time()
            for edge_feature in EDGE_FEATURE:
                train_judge_right_num = 0
                train_ep_num = 0
                for edges in self.edges_list:
                    feature_dict = edges.feature_dict_dict[edge_feature]
                    for ep in edges.edge_pair:
                        train_ep_num += 1
                        judge = judge_by_feature(edge_feature, feature_dict, ep)
                        if judge == edges.edge_pair[ep]:
                            train_judge_right_num += 1
                        elif judge == -1:
                            train_judge_right_num += 0.5
                train_aucs[edge_feature] = train_judge_right_num / train_ep_num
                since = time.time()
                is_big_old = True
                if train_aucs[edge_feature] < 0.5:
                    is_big_old = False
                    train_aucs[edge_feature] = 1 - train_aucs[edge_feature]
                is_big_olds[edge_feature] = is_big_old
            self.best_feature = max(train_aucs, key=train_aucs.get)
            self.is_big_old_for_best_feature = is_big_olds[self.best_feature]
            print("Cross Nets", "best feature: " + self.best_feature, ".is big old: " + str(self.is_big_old_for_best_feature))

        def get_ep_judge_prob(self, edge_pair, embedding):
            """
            :param edge_pair:
            :param embedding: Edges best_feature_dict
            :return:
            """
            # best_feature_dict = edges.feature_dict_dict[self.best_feature]
            bf1, bf2 = embedding[edge_pair[0]], embedding[edge_pair[1]]
            if bf1 == bf2:
                return 0.5
            elif self.is_big_old_for_best_feature:
                return bf2 / (bf1 + bf2)
            else:
                return bf1 / (bf1 + bf2)

        def predict(self, edge_pair, embedding):
            judge = self.get_ep_judge_prob(edge_pair, embedding)
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

    class PairNNJudge:
        def __init__(self, edges_list, embedding_name, save=False, save_path=None, load=False, load_path=None):
            self.embedding_name = embedding_name
            self.edges_list = edges_list
            self.nb_class = 2
            self.train_ep_num = 0
            self.embedding_len = 0
            for edges in self.edges_list:
                self.train_ep_num += len(edges.edge_pair)
            if self.embedding_name == NODE2VEC:
                self.embedding_len = self.edges_list[0].node2vec_embedding.get_embedding_len()
            elif self.embedding_name == COLLECTION:
                self.embedding_len = self.edges_list[0].simple_embedding.get_embedding_len()
            elif self.embedding_name == LINE_EMBEDDING:
                self.embedding_len = self.edges_list[0].line_embedding.get_embedding_len()
            elif self.embedding_name == STRUCT2VEC_EMBEDDING:
                self.embedding_len = self.edges_list[0].struct2vec_embedding.get_embedding_len()
            elif self.embedding_name == DEEPWALK_EMBEDDING:
                self.embedding_len = self.edges_list[0].deepwalk_embedding.get_embedding_len()
            elif self.embedding_name == SDNE_EMBEDDING:
                self.embedding_len = self.edges_list[0].sdne_embedding.get_embedding_len()
            else:
                self.embedding_len = 0
            self.x_edge1_train = None
            self.x_edge2_train = None
            self.y_train = None
            self.graph = None
            self.sess = None
            self.saver = None
            self.checkpoint_path = None
            self.input_edge_1 = None
            self.input_edge_2 = None
            self.logits = None
            self.save = save
            if save_path is not None:
                self.save_path = save_path + "pair_nn/" + embedding_name + "/" + "cpnn"
                if not os.path.exists(save_path + "pair_nn/" + embedding_name + "/"):
                    os.makedirs(save_path + "pair_nn/" + embedding_name + "/")
            self.load = load
            if load_path is not None:
                self.load_path = load_path + "pair_nn/" + embedding_name + "/" + "cpnn"
                if not os.path.exists(load_path + "pair_nn/" + embedding_name + "/"):
                    os.makedirs(load_path + "pair_nn/" + embedding_name + "/")
                if os.path.exists(self.load_path):
                    os.rmdir(self.load_path)
            self.pair_nn_model_train()

        def get_train_data(self):
            self.x_edge1_train = np.zeros([self.train_ep_num, self.embedding_len], dtype=np.float32)
            self.x_edge2_train = np.zeros([self.train_ep_num, self.embedding_len], dtype=np.float32)
            self.y_train = np.zeros([self.train_ep_num, 1], dtype=np.float32)
            i = 0
            for edges in self.edges_list:
                if self.embedding_name == NODE2VEC:
                    embedding = edges.node2vec_embedding
                elif self.embedding_name == COLLECTION:
                    embedding = edges.simple_embedding
                elif self.embedding_name == LINE_EMBEDDING:
                    embedding = edges.line_embedding
                elif self.embedding_name == STRUCT2VEC_EMBEDDING:
                    embedding = edges.struct2vec_embedding
                elif self.embedding_name == DEEPWALK_EMBEDDING:
                    embedding = edges.deepwalk_embedding
                elif self.embedding_name == SDNE_EMBEDDING:
                    embedding = edges.sdne_embedding
                else:
                    embedding = None
                for ep in edges.edge_pair:
                    edge1 = ep[0]
                    edge2 = ep[1]
                    self.x_edge1_train[i, :] = embedding.get_embedding(edge1)  # [n, embedding_len]
                    self.x_edge2_train[i, :] = embedding.get_embedding(edge2)  # [n, embedding_len]
                    self.y_train[i, :] = np.array([edges.edge_pair[ep]])  # [n, 1]
                    i += 1
            self.y_train = np_utils.to_categorical(self.y_train, self.nb_class)
            print("Cross Nets", self.embedding_name, "train edge pair vector get!:")

        def pair_nn_model_train(self):
            single_edge_embedding_len = self.embedding_len
            single_edge_output_len = 1
            single_edge_hidden_layer_len = int(single_edge_embedding_len * 2 / 3) + single_edge_output_len
            self.graph = tf.Graph()
            with self.graph.as_default():
                input_edge_1 = tf.placeholder(tf.float32, [None, single_edge_embedding_len], name="input_edge_1")
                input_edge_2 = tf.placeholder(tf.float32, [None, single_edge_embedding_len], name="input_edge_2")
                output_new_judge = tf.placeholder(tf.float32, [None, 2 * single_edge_output_len],
                                                  name="output_new_judge")
                # layer1
                with tf.name_scope("layer1"):
                    weights_1 = tf.Variable(tf.random_normal([single_edge_embedding_len, single_edge_hidden_layer_len]),
                                            name="weight1", dtype=tf.float32)
                    bias_1 = tf.Variable(tf.random_normal([1, single_edge_hidden_layer_len]), name="bias1", dtype=tf.float32)  # 矩阵相加的广播机制
                    x1w1_plus_b1 = tf.matmul(input_edge_1, weights_1) + bias_1
                    x2w1_plus_b1 = tf.matmul(input_edge_2, weights_1) + bias_1
                    x1_layer1_output = tf.nn.relu(x1w1_plus_b1)
                    x2_layer1_output = tf.nn.relu(x2w1_plus_b1)
                # layer2
                with tf.name_scope("layer2"):
                    weights_2 = tf.Variable(tf.random_normal([single_edge_hidden_layer_len, single_edge_output_len]), 
                        name='weight2', dtype=tf.float32)
                    bias_2 = tf.Variable(tf.random_normal([1, single_edge_output_len]), name="bias2", dtype=tf.float32)
                    x1_layer2_output = tf.matmul(x1_layer1_output, weights_2) + bias_2
                    x2_layer2_output = tf.matmul(x2_layer1_output, weights_2) + bias_2
                regularizer = tf.contrib.layers.l2_regularizer(0.001)
                regularization = regularizer(weights_1) + regularizer(weights_2) + regularizer(bias_1) + regularizer(bias_2)
                tf.add_to_collection("loss", regularization)
                merge_layer2_output = tf.concat([x1_layer2_output, x2_layer2_output], 1,
                                                name="void_edge_time")
                logits = tf.nn.softmax(merge_layer2_output)
                tf.identity(logits, name="logits")
                cross_entrophy = tf.reduce_mean(-tf.reduce_sum(output_new_judge *
                                                               tf.log(tf.clip_by_value(logits, 1e-10, 1.0)),
                                                               reduction_indices=[1]))
                tf.add_to_collection("loss", cross_entrophy)
                with tf.name_scope("loss"):
                    loss = tf.add_n(tf.get_collection("loss"))
                with tf.name_scope("train"):
                    train_step = tf.train.AdamOptimizer(name="adam").minimize(loss)

                tf.summary.scalar("loss", loss)
                # Session
                self.sess = tf.Session(graph=self.graph)
                if self.load and os.path.exists(self.load_path + ".meta"):
                    self.saver = tf.train.Saver()
                    self.saver.restore(self.sess, self.load_path)
                    print(self.load_path, " pair_nn model loaded.")
                else:
                    self.get_train_data()
                    self.sess.run(tf.global_variables_initializer())
                    batch_size = int(self.train_ep_num * 0.01) if int(self.train_ep_num * 0.01) > 0 else 16
                    epoch = 1000
                    loss_val = 0
                    since = time.time()
                    for e in range(epoch):
                        batch_start = 0
                        batch_end = self.train_ep_num if batch_start + batch_size > self.train_ep_num else \
                            batch_start + batch_size
                        while batch_end <= self.train_ep_num:
                            edge1_batch_train = self.x_edge1_train[batch_start:batch_end, :]
                            edge2_batch_train = self.x_edge2_train[batch_start:batch_end, :]
                            y_batch_train = self.y_train[batch_start:batch_end, :]
                            # loss
                            feed_dict = {input_edge_1: edge1_batch_train, input_edge_2: edge2_batch_train,
                                         output_new_judge: y_batch_train}
                            self.sess.run(train_step, feed_dict=feed_dict)
                            if e % 100 == 0 or e == epoch - 1:
                                loss_val += self.sess.run(loss, feed_dict=feed_dict)
                            batch_start = batch_end
                            if batch_end == self.train_ep_num:
                                batch_end = self.train_ep_num + 1
                            else:
                                batch_end = self.train_ep_num if batch_start + batch_size > self.train_ep_num else \
                                    batch_start + batch_size
                        if e % 400 == 0:
                            print("Cross Nets" + " pair_nn train " + "epoch " + str(e), "Loss:" + str(loss_val),
                                  "Time:" + str(time.time() - since))
                            loss_val = 0
                            since = time.time()
                        self.shuffle_train_data()
                    if self.save:
                        self.saver = tf.train.Saver()
                        self.checkpoint_path = self.saver.save(self.sess, self.save_path)
                self.input_edge_1 = self.graph.get_tensor_by_name('input_edge_1:0')
                self.input_edge_2 = self.graph.get_tensor_by_name("input_edge_2:0")
                self.logits = self.graph.get_tensor_by_name("logits:0")
                # self.edge_pair_time = self.graph.get_tensor_by_name("void_edge_time:0")
                del self.x_edge1_train, self.x_edge2_train, self.y_train
                gc.collect()

        def shuffle_train_data(self):
            permutation = np.random.permutation(self.train_ep_num)
            self.x_edge1_train = self.x_edge1_train[permutation, :]
            self.x_edge2_train = self.x_edge2_train[permutation, :]
            self.y_train = self.y_train[permutation, :]

        def get_ep_judge_prob(self, edge_pair, embedding):
            edge1_vec = np.array([embedding.get_embedding(edge_pair[0])])
            edge2_vec = np.array([embedding.get_embedding(edge_pair[1])])
            result = self.sess.run(self.logits, feed_dict={self.input_edge_1: edge1_vec, self.input_edge_2: edge2_vec})
            return result[0, 1]

        def predict(self, edge_pair, embedding):
            judge = self.get_ep_judge_prob(edge_pair, embedding)
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

    def get_test_auc(self):
        print("Test Begin...")
        test_num = 0
        test_right_num = 0
        since = time.time()
        if self.test_nets[0] not in self.train_nets:
            edges_list = [self.Edges(net, self.test_edge_ratio, ensemble=False) for net in self.test_nets]
        else:
            for edges in self.edges_list:
                if edges.net == self.test_nets[0]:
                    edges_list = edges
            edges_list = [edges_list]
        for edges in edges_list:
            for edge_pair, new in getEdgePair.get_single_eps_from_edges(edges.test_edge, edges.net, random_ratio=self.test_ep_ratio):
                test_num += 1
                if self.merge_data_or_model == "data":
                    if self.judge_method == BEST_SINGLE:
                        judge = self.best_feature.predict(edge_pair, edges.feature_dict_dict[self.best_feature.best_feature])
                    if self.judge_method == NODE2VEC_PAIR_NN:
                        judge = self.node2vec.predict(edge_pair, edges.node2vec_embedding)
                    if self.judge_method == UNION_PAIR_NN:
                        judge = self.collection.predict(edge_pair, edges.simple_embedding)
                    if self.judge_method == LINE_PAIR_NN:
                        judge = self.line.predict(edge_pair, edges.line_embedding)
                    if self.judge_method == STRUC2VEC_PAIR_NN:
                        judge = self.struc2vec.predict(edge_pair, edges.struct2vec_embedding)
                    if self.judge_method == DEEPWALK_PAIR_NN:
                        judge = self.deepwalk.predict(edge_pair, edges.deepwalk_embedding)
                    if self.judge_method == SDNE_PAIR_NN:
                        judge = self.sdne.get_ep_judge_prob(edge_pair, edges.sdne_embedding)
                    if self.judge_method == ENSEMBLE:
                        best_feature_judge_prob = \
                            self.best_feature.get_ep_judge_prob(edge_pair, edges.feature_dict_dict[self.best_feature.best_feature])
                        node2vec_judge_prob = self.node2vec.get_ep_judge_prob(edge_pair, edges.node2vec_embedding)
                        collection_judge_prob = self.collection.get_ep_judge_prob(edge_pair, edges.simple_embedding)
                        line_judge_prob = self.line.get_ep_judge_prob(edge_pair, edges.line_embedding)
                        struc2vec_judge_prob = self.struc2vec.get_ep_judge_prob(edge_pair, edges.struct2vec_embedding)
                        deepwalk_judge_prob = self.deepwalk.get_ep_judge_prob(edge_pair, edges.deepwalk_embedding)
                        sdne_judge_prob = self.sdne.get_ep_judge_prob(edge_pair, edges.sdne_embedding)
                        base_model_vec = [best_feature_judge_prob,
                                          node2vec_judge_prob,
                                          collection_judge_prob,
                                          line_judge_prob,
                                          struc2vec_judge_prob,
                                          deepwalk_judge_prob,
                                          sdne_judge_prob]
                        judge = self.clf.predict(base_model_vec)
                elif self.merge_data_or_model == "model":
                    ensemble_judges = []
                    for model in self.models:
                        base_vec = [model[BEST_SINGLE].get_ep_judge_prob(edge_pair, edges), model[NODE2VEC_PAIR_NN].get_ep_judge_prob(edge_pair, edges), model[UNION_PAIR_NN].get_ep_judge_prob(edge_pair, edges)]
                        one_net_ensemble_judge = model[ENSEMBLE].predict(base_vec)
                        ensemble_judges.append(one_net_ensemble_judge)
                    judge = self.clf.predict(ensemble_judges)
                else:
                    judge = None
                if judge == new:
                    test_right_num += 1
                if test_num % 100000 == 0:
                    print("Cross Nets", "Ensemble Model ", "test edge pair num:" + str(test_num),
                          "time spend:" + str(time.time() - since))
                    since = time.time()
        test_auc = test_right_num / test_num
        print("Cross Nets" + " Ensemble Model test ep num:" + str(test_num) + ". Test auc:", test_auc)
        return test_auc


def cross_judge_for_net(test_net, net_names, train_nets_set, judge_method, times, train_edge_ratio,
                        train_net_set_dict, cross_net_auc, file_name, merge_data_or_model, force_calc=False):
    print(test_net, "Train nets set", train_nets_set)
    train_net_set_dict[test_net] = train_nets_set
    for train_nets in train_nets_set:
        for i in range(times):
            if i == len(cross_net_auc):
                cross_net_auc.append(dict())
            if test_net in cross_net_auc[i].keys() and str(train_nets) in cross_net_auc[i][test_net].keys() and force_calc is False:
                print("Test on ", test_net, "Train on ", train_nets, "Times", i, "Done!")
                continue
            aucs = cross_net_auc[i]
            print("Time", i, "Train on", train_nets, "Test on", test_net, "ing...")
            if train_nets[0] == test_net and len(train_nets) == 1:
                cross_net_judge = CrossNetsJudge(train_nets, [test_net], train_edge_ratio, i, judge_method,
                                                 merge_data_or_model)
                test_auc = cross_net_judge.get_test_auc()
                if test_net not in aucs.keys():
                    aucs[test_net] = dict()
                aucs[test_net][str(train_nets)] = test_auc
            else:
                cross_net_judge = CrossNetsJudge(train_nets, [test_net], train_edge_ratio, i, judge_method, merge_data_or_model)
                test_auc = cross_net_judge.get_test_auc()
                if test_net not in aucs.keys():
                    aucs[test_net] = dict()
                aucs[test_net][str(train_nets)] = test_auc
                # pass
            cross_net_auc[i] = aucs
            with open(file_name, 'wb') as f:
                try:
                    pickle.dump(cross_net_auc, f)
                except OSError as reason:
                    print("Error")


def cross_nets_judge(test_nets, train_nets, train_nets_set, judge_method, times, train_edge_ratio,
                     merge_data_or_model="data", force_calc=False):
    """
    Parameters
    ----------
    test_nets
    train_nets
    train_nets_set  ：[[net1], [net2], [net1, net2], ...]
    judge_method    ：BEST_SINGLE, NODE2VEC_PAIR_NN, UNION_PAIR_NN, LINE_PAIR_NN, STRUC2VEC_PAIR_NN,
                    DEEPWALK_PAIR_NN, SDNE_PAIR_NN, ENSEMBLE
    times
    train_edge_ratio
    merge_data_or_model
    Returns
    -------
    """
    print("Train on ", str(train_nets), "Test on ", str(test_nets), "Ratio ", train_edge_ratio, "Begin...")
    file_name = "./judge_data/cross_data/" + str(round(train_edge_ratio, 3)) + judge_method\
                + ",".join(test_nets) + '_' + ",".join(train_nets) + "_7base_train_nets_merge_by_" + merge_data_or_model + "_cross_judge_auc.pkl"
    if os.path.exists(file_name) is False:
        cross_net_auc = list()
        for i in range(times):
            cross_net_auc.append(dict())  # dict[test_net][train_nets]
    else:
        with open(file_name, 'rb') as f:
            cross_net_auc = pickle.load(f)
    train_net_set_dict = dict()
    ps = []
    for test_net in test_nets:
        # ps.append(Process(target=cross_judge_for_net, args=(test_net, net_names, times, train_edge_ratio, train_net_set_dict, cross_net_auc, file_name, merge_data_or_model, )))
        cross_judge_for_net(test_net, train_nets, train_nets_set, judge_method, times, train_edge_ratio, train_net_set_dict,
                            cross_net_auc, file_name, merge_data_or_model, force_calc)
    mean_auc = dict()
    var_auc = dict()
    for test_net in test_nets:
        if test_net not in mean_auc:
            mean_auc[test_net] = dict()
            var_auc[test_net] = dict()
        for train_net_set in train_net_set_dict[test_net]:
            aucs = [a[test_net][str(train_net_set)] for a in cross_net_auc if
                    test_net in a.keys() and str(train_net_set) in a[test_net].keys()]
            mean_auc[test_net][str(train_net_set)] = np.mean(aucs)
            var_auc[test_net][str(train_net_set)] = np.sqrt(np.var(aucs))
    # Excel
    excel_f = xlwt.Workbook()
    sheet = excel_f.add_sheet("cross", cell_overwrite_ok=True)
    for net_name, net_name_i in zip(test_nets, list(range(len(test_nets)))):
        sheet.write_merge(2 * net_name_i + 1, 2 * net_name_i + 2, 0, 0, net_name)
        sheet.write(2 * net_name_i + 1, 1, "accu")
        sheet.write(2 * net_name_i + 2, 1, "std")
        for train_net_set, train_nets_i in zip(train_net_set_dict[net_name], range(len(train_net_set_dict[net_name]))):
            sheet.write(0, train_nets_i + 2, str(train_net_set))
            sheet.write(2 * net_name_i + 1, train_nets_i + 2, mean_auc[net_name][str(train_net_set)])
            sheet.write(2 * net_name_i + 2, train_nets_i + 2, var_auc[net_name][str(train_net_set)])
    excelFileName = "./Result/Cross_Nets/DirectTransfer/" +"DirectTransfer"+ str(round(train_edge_ratio, 3)) + judge_method \
                    + ",".join(test_nets) + '_' + ",".join(train_nets) + "_7_base_train_nets_merge_by_" + \
                    merge_data_or_model + "_cross_nets_accu.xls"
    excel_f.save(excelFileName)


if __name__ == '__main__':
    times = 10
    train_edge_ratio = 0.4
    test_nets = ["BA13%", "BA15%", "PSO21%", "PSO22%", "Fitness14%", "Fitness15%"]
    train_nets = ["BA13%", "BA15%", "PSO21%", "PSO22%", "Fitness14%", "Fitness15%"]
    train_nets_set = [[name] for name in train_nets]
    judge_method = ENSEMBLE
    force_calc = True
    cross_nets_judge(test_nets, train_nets, train_nets_set, judge_method, times, train_edge_ratio,
                     merge_data_or_model="data", force_calc=force_calc)
