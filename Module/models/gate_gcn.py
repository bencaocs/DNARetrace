import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

import layers


class GraphGatedGCNModel(nn.Module):
    def __init__(self, node_features=1, edge_features=1, hidden_features=32, hidden_edge_features=32, num_layers=4, hidden_edge_scores=32, batch_norm=True):
        super().__init__()
        #self.node_encoder = layers.NodeEncoder(node_features, hidden_features)
        self.linear_pe = nn.Linear(node_features, hidden_features) # PE + degree_in + degree_out 
        #self.edge_encoder = layers.EdgeEncoder(edge_features, hidden_features)
        self.linear_edge = nn.Linear(edge_features, hidden_features) 
        # self.linear2_edge = nn.Linear(hidden_edge_features, hidden_features) 
        self.gnn = layers.GraphGatedGCN(num_layers, hidden_features, batch_norm)
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)

    def forward(self, graph, x, e):
        x = self.linear_pe(x) 
        e = self.linear_edge(e)
        # e = torch.relu(e)
        # e = self.linear2_edge(e)
        x, e = self.gnn(graph, x, e)
        scores = self.predictor(graph, x, e)
        return scores

