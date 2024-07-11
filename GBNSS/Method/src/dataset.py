#%%
import numpy as np
import os
import networkx as nx
from tqdm import tqdm
import pandas as pd
import torch
#%%
class BiologicalDataset(object):
    def __init__(self, root, test=False):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir(download dataset) and processed_dir(processed data).
        """
        self.test = test
        self.raw_paths = os.path.join(root)
        
    def process(self):
        """
        Processing the test/train txt file into a list contained dicts
        """
        datalist = []
        self.init_data = pd.read_csv(self.raw_paths, sep="\t", names=["graph1", "graph2", "ec", "lccs", "s3", "go"])
        for index, pair in tqdm(self.init_data.iterrows(), total=self.init_data.shape[0]):
            graph_1, feature_1 = self._get_adjacency_info(pair["graph1"])
            graph_2, feature_2 = self._get_adjacency_info(pair["graph2"])
            go = pair["go"]

            # Creating data dict
            data = {"graph_1": graph_1,
                    "feature_1": feature_1,
                    "graph_2": graph_2,
                    "feature_2": feature_2,
                    "go": go
                    }
            new_data = self.data_process(data)
            datalist.append(new_data)
        return datalist

    def data_process(self, data):
        new_data = dict()
        edges_1 = data["graph_1"] + [[y, x] for x, y in data["graph_1"]]
        edges_2 = data["graph_2"] + [[y, x] for x, y in data["graph_2"]]

        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)

        features_1 = []
        features_2 = []

        for n in data["feature_1"]:
            features_1.append(n)

        for n in data["feature_2"]:
            features_2.append(n)

        features_1 = torch.FloatTensor(np.array(features_1))
        features_2 = torch.FloatTensor(np.array(features_2))

        new_data["edge_index_1"] = edges_1
        new_data["edge_index_2"] = edges_2

        new_data["features_1"] = features_1
        new_data["features_2"] = features_2

        norm_go = data["go"]/(0.5*(len(data["feature_1"])+len(data["feature_2"])))
        new_data["go"] = torch.from_numpy(np.exp(-norm_go).reshape(1, 1)).view(-1).float()
        return new_data

    def _get_adjacency_info(self, graph):
        #构造节点索引
        graph_path = f"./../dataset/rawData/{graph}"
        graphs = nx.read_edgelist(graph_path)
        node = graphs.nodes
        node_index = {node:index for index, node in enumerate(node)}
        
        #获得边信息
        edges = np.array(graphs.edges)
        edgelist = []

        for edge in edges:
            edgelist.append([node_index[edge[0]], node_index[edge[1]]])
        

        #获得节点特征信息
        features = []
        feature_path = "./../dataset/bio_feature.csv"
        feature_table = pd.read_csv(feature_path, sep="\t")
        for node in node_index:
            features.append(feature_table.loc[feature_table.entry == node].values[0][1:])
        features = np.array(features, dtype=np.float32).reshape(-1, 3)
        return edgelist, features

class TopologicalDataset(object):
    def __init__(self, root, test=False):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir(download dataset) and processed_dir(processed data).
        """
        self.test = test
        self.raw_paths = os.path.join(root)
  
    def process(self):
        """
        Processing the test/train txt file into a list contained dicts
        """
        datalits = []
        self.init_data = pd.read_csv(self.raw_paths, sep="\t", names=["graph1", "graph2", "ec", "lccs", "s3", "go"])
        for index, pair in tqdm(self.init_data.iterrows(), total=self.init_data.shape[0]):
            graph_1, feature_1 = self._get_adjacency_info(pair["graph1"])
            graph_2, feature_2 = self._get_adjacency_info(pair["graph2"])
            ec = pair["ec"]
            lccs = pair["lccs"]
            s3 = pair["s3"]

            # Create data dict
            data = {"graph_1": graph_1,
                    "features_1": feature_1,
                    "graph_2": graph_2,
                    "features_2": feature_2,
                    "ec": ec,
                    "lccs": lccs,
                    "s3": s3
                    }
            new_data = self.data_process(data)
            datalits.append(new_data)
        return datalits

    def data_process(self, data):
        new_data = dict()
        edges_1 = data["graph_1"] + [[y, x] for x, y in data["graph_1"]]
        edges_2 = data["graph_2"] + [[y, x] for x, y in data["graph_2"]]

        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)

        features_1 = []
        features_2 = []

        for n in data["features_1"]:
            features_1.append(n)

        for n in data["features_2"]:
            features_2.append(n)

        features_1 = torch.FloatTensor(np.array(features_1))
        features_2 = torch.FloatTensor(np.array(features_2))

        new_data["edge_index_1"] = edges_1
        new_data["edge_index_2"] = edges_2

        new_data["features_1"] = features_1
        new_data["features_2"] = features_2

        norm_lccs = data["lccs"]/(0.5*(len(data["features_1"])+len(data["features_2"])))
        new_data["lccs"] = torch.from_numpy(np.exp(-norm_lccs).reshape(1, 1)).view(-1).float()
        new_data["ec"] = torch.tensor(data["ec"])
        new_data["s3"] = torch.tensor(data["s3"])
        
        return new_data

    def _get_adjacency_info(self, graph):
        #构造节点索引
        graph_path = f"./../dataset/rawData/{graph}"
        graphs = nx.read_edgelist(graph_path)
        node = graphs.nodes
        node_index = {node:index for index, node in enumerate(node)}
        
        #获得边信息
        edges = np.array(graphs.edges)
        edgelist = []

        for edge in edges:
            edgelist.append([node_index[edge[0]], node_index[edge[1]]])
        

        #获得节点特征信息
        features = []
        feature_path = "./../dataset/topo_feature.csv"
        feature_table = pd.read_csv(feature_path, sep="\t")
        for node in node_index:
            features.append(feature_table.loc[feature_table.entry == node].values[0][1:])
        features = np.array(features, dtype=np.float32).reshape(-1, 81)
        return edgelist, features