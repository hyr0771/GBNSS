#%%
import pandas as pd
import torch
from dataset import BiologicalDataset, TopologicalDataset
import numpy as np
import networkx as nx
import os
from model import GBNSS
from param_parser import parameter_parser

#%%
args = parameter_parser()

#%%
def get_adjacency(graph):
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
    go_features = []
    to_features = []
    go_feature_path = "./../dataset/bio_feature.csv"
    to_feature_path = "./../dataset/topo_feature.csv"
    go_feature_table = pd.read_csv(go_feature_path, sep="\t")
    for node in node_index:
        go_features.append(go_feature_table.loc[go_feature_table.entry == node].values[0][1:])
    go_features = np.array(go_features, dtype=np.float32).reshape(-1, 3)

    to_feature_table = pd.read_csv(to_feature_path, sep="\t")
    for node in node_index:
        to_features.append(to_feature_table.loc[to_feature_table.entry == node].values[0][1:])
    to_features = np.array(to_features, dtype=np.float32).reshape(-1, 3)

    return edgelist, go_features, to_features 

def go_data_prepare(query, db):
    data = dict()
    query_edges, query_go_features, query_to_features = get_adjacency(query)
    db_edges, db_go_features, db_to_features = get_adjacency(db)
    data = {
        "graph_1": query_edges,
        "feature_1": query_go_features,
        "graph_2": db_edges,
        "feature_2": db_go_features,
        }

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
    
    return new_data
        
def to_data_prepare(query, db):
    data = dict()
    query_edges, query_go_features, query_to_features = get_adjacency(query)
    db_edges, db_go_features, db_to_features = get_adjacency(db)

    data = {
        "graph_1": query_edges,
        "feature_1": query_to_features,
        "graph_2": db_edges,
        "feature_2": db_to_features,
        }

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
    
    return new_data

#%%
file_list = os.listdir("./../dataset/rawData")

db = file_list[:70]
query = file_list[70:]
data = go_data_prepare(query[0], db[0])

model = GBNSS(args, features_size=data["features_1"].shape[1])
model.load_state_dict(torch.load("./../save_model/gbnss_go.pth"))
result = model(data).item()

result = []
for query_file in query:
    for db_file in db:
        data = go_data_prepare(query_file, db_file)
        pred = model(data).item()
        result.append({"graph1":query_file, "graph2":db_file, "results":pred})
df = pd.DataFrame(result)
df = df.sort_values(by=(["graph1", "results"]), ascending=False)    
df.to_csv("./../Result/result.csv", sep="\t", index=None)
