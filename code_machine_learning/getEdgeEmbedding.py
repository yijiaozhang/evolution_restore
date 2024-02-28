import numpy as np
import os
import pickle
import networkx as nx
from getNet import get_mat, get_adj
from getFeatureJudge import get_feature_dict
from __Configuration import *
from node2vec.main import my_main
from line.line import LINE
from struct2vec.struc2vec import Struc2Vec
from deepwalk.deepwalk import DeepWalk
from sdne.sdne import SDNE


class SingleEmbedding:
    def __init__(self, net_name, feature_name):
        self.net_name = net_name
        self.feature_dict = dict()
        self.min_feature = 0
        self.max_feature = 0
        net_mat = get_mat(net_name)
        self.feature_dict = get_feature_dict(net_mat, feature_name, net_name)
        self.min_feature = np.min(np.array(list(self.feature_dict.values())))
        self.max_feature = np.max(np.array(list(self.feature_dict.values())))

    def get_embedding_name(self):
        return SINGLE

    def get_embedding_len(self):
        return 1

    def get_embedding(self, edge):
        edge_vec = []
        feature_dict = self.feature_dict
        if self.max_feature - self.min_feature != 0:
            edge_vec.append((feature_dict[edge] - self.min_feature) / (self.max_feature - self.min_feature))  # min_max
        else:
            edge_vec.append(0)
        return edge_vec


class SimpleEmbedding:
    def __init__(self, net_name):
        self.net_name = net_name
        self.feature_dict_dict = dict()
        self.min_feature = dict()
        self.max_feature = dict()
        net_mat = get_mat(net_name)
        for judge_method in EDGE_FEATURE:
            self.feature_dict_dict[judge_method] = get_feature_dict(net_mat, judge_method, net_name)
        for judge_method in EDGE_FEATURE:
            self.min_feature[judge_method] = np.min(np.array(list(self.feature_dict_dict[judge_method].values())))
            self.max_feature[judge_method] = np.max(np.array(list(self.feature_dict_dict[judge_method].values())))

    def get_embedding_name(self):
        return COLLECTION

    def get_embedding_len(self):
        return len(EDGE_FEATURE)

    def get_embedding(self, edge):
        edge_vec = []
        for feature in EDGE_FEATURE:
            feature_dict = self.feature_dict_dict[feature]
            if self.max_feature[feature] - self.min_feature[feature] != 0:
                edge_vec.append((feature_dict[edge] - self.min_feature[feature]) / (self.max_feature[feature] - self.min_feature[feature]))
            else:
                edge_vec.append(0)
        return edge_vec


class Node2vecEmbedding:
    def __init__(self, net_name, edges=None, force_calc=False, node_embedding_path=None, edge_embedding_path=None):
        self.net_name = net_name
        self.edges = get_adj(self.net_name) if edges is None else edges
        self.force_calc = force_calc
        self.node_embedding_path = "./node2vec/" + str(self.net_name) + ".emd" if node_embedding_path is None else node_embedding_path
        self.edge_embedding_path = "./edge_node2vec-hadamard-edge2vec/" + self.net_name + "edge2vec.pkl" if edge_embedding_path is None else edge_embedding_path
        self.node_vec_dict, self.node_vec_len = self.get_node2vec()
        self.edge_vecs = self.get_edge2vecs(self.node_vec_dict)

    def get_node2vec(self):
        filename = self.node_embedding_path
        if os.path.exists(filename) and self.force_calc is False:
            pass
        else:
            print(self.net_name, "node2vec running...")
            my_main(self.edges, filename)
        with open(filename, 'r') as fn:
            line = fn.readline()
            line = line.strip()
            s = line.split(' ')
            node_vec_len = int(s[1])
            node_vec = dict()
            line = fn.readline()
            while line:
                s = line.strip().split(' ')
                node_vec[int(s[0])] = [float(x) for x in s[1:]]
                line = fn.readline()
        return node_vec, node_vec_len

    def get_edge2vecs(self, node_vec_dict):
        file_name = self.edge_embedding_path
        if os.path.exists(file_name) is False or self.force_calc:
            edge_vecs = dict()
            for edge in self.edges:
                node1vec = node_vec_dict[edge[0]]
                node2vec = node_vec_dict[edge[1]]
                edge_vec = np.array(node1vec) * np.array(node2vec)
                edge_vec = list(edge_vec)
                edge_vecs[edge] = edge_vec
            with open(file_name, 'wb') as f:
                pickle.dump(edge_vecs, f)
        else:
            with open(file_name, 'rb') as f:
                edge_vecs = pickle.load(f)
        return edge_vecs

    def get_embedding_name(self):
        return NODE2VEC

    def get_embedding_len(self):
        return self.node_vec_len

    def get_embedding(self, edge):
        return self.edge_vecs[edge]


class LineEmbedding:
    def __init__(self, net_name, edges=None, force_calc=False, node_embedding_path=None, edge_embedding_path=None):
        self.net_name = net_name
        self.edges = get_adj(self.net_name) if edges is None else edges
        self.force_calc = force_calc
        self.node_embedding_path = "./line/" + str(self.net_name) + ".emd" if node_embedding_path is None else node_embedding_path
        self.edge_embedding_path = "./edge_line-hadamard-edge2vec/" + self.net_name + "edge2vec.pkl" if edge_embedding_path is None else edge_embedding_path
        self.node_vec_dict, self.node_vec_len = self.get_node2vec()
        self.edge_vecs = self.get_edge2vecs(self.node_vec_dict)

    def get_node2vec(self):
        filename = self.node_embedding_path
        if os.path.exists(filename) and self.force_calc is False:
            pass
        else:
            print(self.net_name, "line running...")
            nx_G = nx.DiGraph()
            nx_G.add_edges_from(self.edges)
            for edge in nx_G.edges():
                nx_G[edge[0]][edge[1]]['weight'] = 1
            nx_G = nx_G.to_undirected()
            model = LINE(nx_G, embedding_size=128, order='second')
            model.train(batch_size=1024, epochs=50, verbose=2)
            embeddings = model.get_embeddings()
            with open(filename, 'w') as fw:
                fw.write(str(len(embeddings)) + " " + "128\n")
                for node in embeddings.keys():
                    fw.write(str(node) + " " + " ".join([str(i) for i in embeddings[node]]) + "\n")
        with open(filename, 'r') as fn:
            # get node_vec_len
            line = fn.readline()
            line = line.strip()
            s = line.split(' ')
            node_vec_len = int(s[1])
            # get node_vec
            node_vec = dict()
            line = fn.readline()
            while line:
                s = line.strip().split(' ')
                node_vec[int(s[0])] = [float(x) for x in s[1:]]
                line = fn.readline()
        return node_vec, node_vec_len

    def get_edge2vecs(self, node_vec_dict):
        file_name = self.edge_embedding_path
        if os.path.exists(file_name) is False or self.force_calc:
            edge_vecs = dict()
            for edge in self.edges:
                node1vec = node_vec_dict[edge[0]]
                node2vec = node_vec_dict[edge[1]]
                edge_vec = np.array(node1vec) * np.array(node2vec)
                edge_vec = list(edge_vec)
                edge_vecs[edge] = edge_vec
            with open(file_name, 'wb') as f:
                pickle.dump(edge_vecs, f)
        else:
            with open(file_name, 'rb') as f:
                edge_vecs = pickle.load(f)
        return edge_vecs

    def get_embedding_name(self):
        return LINE_EMBEDDING

    def get_embedding_len(self):
        return self.node_vec_len

    def get_embedding(self, edge):
        return self.edge_vecs[edge]


class Struct2vecEmbedding:
    def __init__(self, net_name, edges=None, force_calc=False, node_embedding_path=None, edge_embedding_path=None):
        self.net_name = net_name
        self.edges = get_adj(self.net_name) if edges is None else edges
        self.force_calc = force_calc
        self.node_embedding_path = "./struct2vec/" + str(self.net_name) + ".emd" if node_embedding_path is None else node_embedding_path
        self.edge_embedding_path = "./edge_struct2vec-hadamard-edge2vec/" + self.net_name + "edge2vec.pkl" if edge_embedding_path is None else edge_embedding_path
        self.node_vec_dict, self.node_vec_len = self.get_node2vec()
        self.edge_vecs = self.get_edge2vecs(self.node_vec_dict)

    def get_node2vec(self):
        filename = self.node_embedding_path
        if os.path.exists(filename) and self.force_calc is False:
            pass
        else:
            print(self.net_name, "struct2vec running...")
            nx_G = nx.DiGraph()
            nx_G.add_edges_from(self.edges)
            for edge in nx_G.edges():
                nx_G[edge[0]][edge[1]]['weight'] = 1
            nx_G = nx_G.to_undirected()

            model = Struc2Vec(nx_G, 10, 80, workers=4, verbose=40, )
            model.train(embed_size=128)
            embeddings = model.get_embeddings()
            with open(filename, 'w') as fw:
                fw.write(str(len(embeddings)) + " " + "128\n")
                for node in embeddings.keys():
                    fw.write(str(node) + " " + " ".join([str(i) for i in embeddings[node]]) + "\n")
        with open(filename, 'r') as fn:
            # get node_vec_len
            line = fn.readline()
            line = line.strip()
            s = line.split(' ')
            node_vec_len = int(s[1])
            # get node_vec
            node_vec = dict()
            line = fn.readline()
            while line:
                s = line.strip().split(' ')
                node_vec[int(s[0])] = [float(x) for x in s[1:]]
                line = fn.readline()
        return node_vec, node_vec_len
 
    def get_edge2vecs(self, node_vec_dict):
        file_name = self.edge_embedding_path
        if os.path.exists(file_name) is False or self.force_calc:
            edge_vecs = dict()
            for edge in self.edges:
                node1vec = node_vec_dict[edge[0]]
                node2vec = node_vec_dict[edge[1]]
                edge_vec = np.array(node1vec) * np.array(node2vec)
                edge_vec = list(edge_vec)
                edge_vecs[edge] = edge_vec
            with open(file_name, 'wb') as f:
                pickle.dump(edge_vecs, f)
        else:
            with open(file_name, 'rb') as f:
                edge_vecs = pickle.load(f)
        return edge_vecs

    def get_embedding_name(self):
        return STRUCT2VEC_EMBEDDING

    def get_embedding_len(self):
        return self.node_vec_len

    def get_embedding(self, edge):
        return self.edge_vecs[edge]


class DeepwalkEmbedding:
    def __init__(self, net_name, edges=None, force_calc=False, node_embedding_path=None, edge_embedding_path=None):
        self.net_name = net_name
        self.edges = get_adj(self.net_name) if edges is None else edges
        self.force_calc = force_calc
        self.node_embedding_path = "./deepwalk/" + str(self.net_name) + ".emd" if node_embedding_path is None else node_embedding_path
        self.edge_embedding_path = "./edge_deepwalk-hadamard-edge2vec/" + self.net_name + "edge2vec.pkl" if edge_embedding_path is None else edge_embedding_path
        self.node_vec_dict, self.node_vec_len = self.get_node2vec()
        self.edge_vecs = self.get_edge2vecs(self.node_vec_dict)

    def get_node2vec(self):
        filename = self.node_embedding_path
        if os.path.exists(filename) and self.force_calc is False:
            pass
        else:
            print(self.net_name, "deepwalk running...")
            nx_G = nx.DiGraph()
            nx_G.add_edges_from(self.edges)
            for edge in nx_G.edges():
                nx_G[edge[0]][edge[1]]['weight'] = 1
            nx_G = nx_G.to_undirected()

            model = DeepWalk(nx_G,walk_length=10,num_walks=80,workers=1)
            model.train(embed_size=128)
            embeddings = model.get_embeddings()
            with open(filename, 'w') as fw:
                fw.write(str(len(embeddings)) + " " + "128\n")
                for node in embeddings.keys():
                    fw.write(str(node) + " " + " ".join([str(i) for i in embeddings[node]]) + "\n")
        with open(filename, 'r') as fn:
            # get node_vec_len
            line = fn.readline()
            line = line.strip()
            s = line.split(' ')
            node_vec_len = int(s[1])
            # get node_vec
            node_vec = dict()
            line = fn.readline()
            while line:
                s = line.strip().split(' ')
                node_vec[int(s[0])] = [float(x) for x in s[1:]]
                line = fn.readline()
        return node_vec, node_vec_len

    def get_edge2vecs(self, node_vec_dict):
        file_name = self.edge_embedding_path
        if os.path.exists(file_name) is False or self.force_calc:
            edge_vecs = dict()
            for edge in self.edges:
                node1vec = node_vec_dict[edge[0]]
                node2vec = node_vec_dict[edge[1]]
                edge_vec = np.array(node1vec) * np.array(node2vec)
                edge_vec = list(edge_vec)
                edge_vecs[edge] = edge_vec
            with open(file_name, 'wb') as f:
                pickle.dump(edge_vecs, f)
        else:
            with open(file_name, 'rb') as f:
                edge_vecs = pickle.load(f)
        return edge_vecs

    def get_embedding_name(self):
        return DEEPWALK_EMBEDDING

    def get_embedding_len(self):
        return self.node_vec_len

    def get_embedding(self, edge):
        return self.edge_vecs[edge]


class SdneEmbedding:
    def __init__(self, net_name, edges=None, force_calc=False, node_embedding_path=None, edge_embedding_path=None):
        self.net_name = net_name
        self.edges = get_adj(self.net_name) if edges is None else edges
        self.force_calc = force_calc
        self.node_embedding_path = "./sdne/" + str(self.net_name) + ".emd" if node_embedding_path is None else node_embedding_path
        self.edge_embedding_path = "./edge_sdne-hadamard-edge2vec/" + self.net_name + "edge2vec.pkl" if edge_embedding_path is None else edge_embedding_path
        self.node_vec_dict, self.node_vec_len = self.get_node2vec()
        self.edge_vecs = self.get_edge2vecs(self.node_vec_dict)

    def get_node2vec(self):
        filename = self.node_embedding_path
        if os.path.exists(filename) and self.force_calc is False:
            pass
        else:
            print(self.net_name, "sdne running...")
            nx_G = nx.DiGraph()
            nx_G.add_edges_from(self.edges)
            for edge in nx_G.edges():
                nx_G[edge[0]][edge[1]]['weight'] = 1
            nx_G = nx_G.to_undirected()

            model = SDNE(nx_G, hidden_size=[256, 128])
            model.train(batch_size=3000,epochs=40,verbose=2)
            embeddings = model.get_embeddings()
            with open(filename, 'w') as fw:
                fw.write(str(len(embeddings)) + " " + "128\n")
                for node in embeddings.keys():
                    fw.write(str(node) + " " + " ".join([str(i) for i in embeddings[node]]) + "\n")
        with open(filename, 'r') as fn:
            # get node_vec_len
            line = fn.readline()
            line = line.strip()
            s = line.split(' ')
            node_vec_len = int(s[1])
            # get node_vec
            node_vec = dict()
            line = fn.readline()
            while line:
                s = line.strip().split(' ')
                node_vec[int(s[0])] = [float(x) for x in s[1:]]
                line = fn.readline()
        return node_vec, node_vec_len

    def get_edge2vecs(self, node_vec_dict):
        file_name = self.edge_embedding_path
        if os.path.exists(file_name) is False or self.force_calc:
            edge_vecs = dict()
            for edge in self.edges:
                node1vec = node_vec_dict[edge[0]]
                node2vec = node_vec_dict[edge[1]]
                edge_vec = np.array(node1vec) * np.array(node2vec)
                edge_vec = list(edge_vec)
                edge_vecs[edge] = edge_vec
            with open(file_name, 'wb') as f:
                pickle.dump(edge_vecs, f)
        else:
            with open(file_name, 'rb') as f:
                edge_vecs = pickle.load(f)
        return edge_vecs

    def get_embedding_name(self):
        return SDNE_EMBEDDING

    def get_embedding_len(self):
        return self.node_vec_len

    def get_embedding(self, edge):
        return self.edge_vecs[edge]


if __name__ == '__main__':
    # TEST
    m = SdneEmbedding("fungi_mirrorTn%4")
