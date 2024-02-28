import numpy as np
import random
import os
import getEdgePair
import tensorflow as tf
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras import layers, losses
import time
import gc
from __Configuration import *
from getEdgeEmbedding import *


class PairStructureNNJudge:

    def __init__(self, net_name, train_eps, train_news, embedding, train_edge_ratio=0.4,
                 save=False, save_path=None, load=False, load_path=None, version="tensorflow"):
        self.net_name = net_name
        self.train_eps = train_eps
        self.train_news = train_news
        self.embedding = embedding
        # Train data
        self.train_ep_num = len(self.train_eps) if train_eps is not None else 0
        self.x_edge1_train = None
        self.x_edge2_train = None
        self.y_train = None
        self.nb_class = 2
        self.version = version
        if save_path is not None and not os.path.exists(save_path + "pair_nn/" + embedding.get_embedding_name() + "/"):
            os.makedirs(save_path + "pair_nn/" + embedding.get_embedding_name() + "/")
        if save_path is not None:
            self.save_path = save_path + "pair_nn/" + embedding.get_embedding_name() + "/" + "cpnn"
        if load_path is not None and not os.path.exists(load_path + "pair_nn/" + embedding.get_embedding_name() + "/"):
            os.makedirs(load_path + "pair_nn/" + embedding.get_embedding_name() + "/")
        if load_path is not None:
            self.load_path = load_path + "pair_nn/" + embedding.get_embedding_name() + "/" + "cpnn"
        self.save = save
        self.load = load

        if version == "tensorflow":
            # Network Need
            self.graph = None
            self.sess = None
            self.saver = None
            self.checkpoint_path = None
            self.input_edge_1 = None
            self.input_edge_2 = None
            self.logits = None
            self.edge_pair_time = None
            self.pair_nn_model_train()
        elif version == "torch":
            self.net = None
            self.pair_nn_model_train_pytorch()


    def get_train_data(self):
        train_label1_num = np.sum(np.array(self.train_news))
        self.x_edge1_train = np.zeros([self.train_ep_num, self.embedding.get_embedding_len()], dtype="float32")
        self.x_edge2_train = np.zeros([self.train_ep_num, self.embedding.get_embedding_len()], dtype="float32")
        self.y_train = np.zeros([self.train_ep_num, 1], dtype="float32")
        index = list(range(self.train_ep_num))
        for edge_pair, new, i in zip(self.train_eps, self.train_news, index):
            edge1 = edge_pair[0]
            edge2 = edge_pair[1]
            self.x_edge1_train[i, :] = self.embedding.get_embedding(edge1)  # [n, embedding_len]
            self.x_edge2_train[i, :] = self.embedding.get_embedding(edge2)  # [n, embedding_len]
            self.y_train[i, :] = np.array([self.train_news[i]], dtype="float32")  # [n, 1]
        if self.version == 'tensorflow':
            self.y_train = np_utils.to_categorical(self.y_train, self.nb_class)
        # print(self.net_name, self.embedding.get_embedding_name(), "train edge pair vector get!:")

    def pair_nn_model_train(self):
        single_edge_embedding_len = self.embedding.get_embedding_len()
        single_edge_output_len = 1
        single_edge_hidden_layer_len = int(single_edge_embedding_len * 2/3) + single_edge_output_len  # cpnn输入层的维数
        self.graph = tf.Graph()
        with self.graph.as_default():
            input_edge_1 = tf.placeholder(tf.float32, [None, single_edge_embedding_len], name="input_edge_1")
            input_edge_2 = tf.placeholder(tf.float32, [None, single_edge_embedding_len], name="input_edge_2")
            output_new_judge = tf.placeholder(tf.float32, [None, 2 * single_edge_output_len],
                                              name="output_new_judge")
            with tf.name_scope("layer1"):
                weights_1 = tf.Variable(tf.random_normal([single_edge_embedding_len, single_edge_hidden_layer_len]),
                                        name="weight1")
                bias_1 = tf.Variable(tf.random_normal([1, single_edge_hidden_layer_len]), name="bias1")
                x1w1_plus_b1 = tf.matmul(input_edge_1, weights_1) + bias_1
                x2w1_plus_b1 = tf.matmul(input_edge_2, weights_1) + bias_1
                x1_layer1_output = tf.nn.relu(x1w1_plus_b1)
                x2_layer1_output = tf.nn.relu(x2w1_plus_b1)
            with tf.name_scope("layer2"):
                weights_2 = tf.Variable(tf.random_normal([single_edge_hidden_layer_len, single_edge_output_len],
                                                         name='weight2'))
                bias_2 = tf.Variable(tf.random_normal([1, single_edge_output_len]), name="bias2")
                x1_layer2_output = tf.matmul(x1_layer1_output, weights_2) + bias_2
                x2_layer2_output = tf.matmul(x2_layer1_output, weights_2) + bias_2
            regularizer = tf.contrib.layers.l2_regularizer(0.001)
            regularization = regularizer(weights_1) + regularizer(weights_2) + regularizer(bias_1) + regularizer(bias_2)
            tf.add_to_collection("loss", regularization)
            # loss function
            merge_layer2_output = tf.concat([x1_layer2_output, x2_layer2_output], 1, name="void_edge_time")
            logits = tf.nn.softmax(merge_layer2_output)
            tf.identity(logits, name="logits")
            cross_entrophy = tf.reduce_mean(-tf.reduce_sum(output_new_judge * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)), reduction_indices=[1]))
            tf.add_to_collection("loss", cross_entrophy)
            with tf.name_scope("loss"):
                loss = tf.add_n(tf.get_collection("loss"))
            with tf.name_scope("train"):
                train_step = tf.train.AdamOptimizer().minimize(loss)
            tf.summary.scalar("loss", loss)
            # Session
            self.sess = tf.Session(graph=self.graph)
            if self.load and os.path.exists(self.load_path + ".meta"):
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, self.load_path)
                print(self.net_name, self.embedding.get_embedding_name(), " pair_nn model loaded.")
            else:
                if self.train_eps is not None and self.train_news is not None:
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
                        feed_dict = {input_edge_1: edge1_batch_train, input_edge_2: edge2_batch_train,
                                     output_new_judge: y_batch_train}
                        self.sess.run(train_step, feed_dict=feed_dict)
                        if e % 100 == 0 or e == epoch-1:
                            loss_val += self.sess.run(loss, feed_dict=feed_dict)
                        batch_start = batch_end
                        if batch_end == self.train_ep_num:
                            batch_end = self.train_ep_num + 1
                        else:
                            batch_end = self.train_ep_num if batch_start + batch_size > self.train_ep_num else \
                                batch_start + batch_size
                    if e % 400 == 0 or e == epoch - 1:
                        print(self.net_name + " pair_nn train " + "epoch " + str(e), "Loss:" + str(loss_val),
                              "Time:" + str(time.time()-since))
                        loss_val = 0
                        since = time.time()
                    self.shuffle_train_data()
                # print(self.net_name + "train " + "epoch " + str(e), "Loss:" + str(loss_val))
                if self.save:
                    self.saver = tf.train.Saver()
                    self.checkpoint_path = self.saver.save(self.sess, self.save_path)
            self.input_edge_1 = self.graph.get_tensor_by_name('input_edge_1:0')
            self.input_edge_2 = self.graph.get_tensor_by_name("input_edge_2:0")
            self.logits = self.graph.get_tensor_by_name("logits:0")
            self.edge_pair_time = self.graph.get_tensor_by_name("void_edge_time:0")
            del self.x_edge1_train, self.x_edge2_train, self.y_train
            gc.collect()

    def shuffle_train_data(self):
        permutation = np.random.permutation(self.train_ep_num)
        self.x_edge1_train = self.x_edge1_train[permutation, :]
        self.x_edge2_train = self.x_edge2_train[permutation, :]
        self.y_train = self.y_train[permutation, :]

    def get_ep_judge_prob(self, edge_pair, feature_dict_dict=None):
        if feature_dict_dict is None:
            edge1_vec = np.array([self.embedding.get_embedding(edge_pair[0])])
            edge2_vec = np.array([self.embedding.get_embedding(edge_pair[1])])
        else:
            edge1_vec = feature_dict_dict[self.embedding.get_embedding_name()][edge_pair[0]]
            edge2_vec = feature_dict_dict[self.embedding.get_embedding_name()][edge_pair[1]]
            edge1_vec = np.array([edge1_vec])
            edge2_vec = np.array([edge2_vec])
        if self.version == "tensorflow":
            result = self.sess.run(self.logits, feed_dict={self.input_edge_1: edge1_vec, self.input_edge_2: edge2_vec})
            return result[0, 1]

    def __del__(self):
        self.sess.close()
