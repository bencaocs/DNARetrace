import torch
import torch.nn as nn
import torch.nn.functional as F
import layers
from utils import *

class GATmodel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_features, 
                num_layers, hidden_edge_scores,
                batch_norm, nb_pos_enc, dropout, nheads, alpha_gat):
        super().__init__()
        
        self.linear1_node = nn.Linear(2, hidden_features) 
        self.linear2_node = nn.Linear(hidden_features*nheads, hidden_features) 

        self.linear1_edge = nn.Linear(edge_features, hidden_features) 
        self.linear2_edge = nn.Linear(hidden_features*nheads, hidden_features)
        
        self.predictor = layers.ScorePredictor(hidden_features, hidden_edge_scores)  

        self.dropout = dropout

        # 多层多头注意力机制
        self.attention_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleList([layers.GATLayer(hidden_features, dropout, alpha_gat, batch_norm, concat=True) 
                                for _ in range(nheads)])
            self.attention_layers.append(layer)

        self.hiddens = [node_features + 2, nb_pos_enc, nb_pos_enc, nb_pos_enc]
        self.denses = nn.ModuleList([nn.Linear(in_features, out_features) for in_features, out_features in zip(self.hiddens[:-1], self.hiddens[1:])])

        # self.centrality_encoding = layers.CentralityEncoding(
        #     max_in_degree=32,#32
        #     max_out_degree=32,
        #     node_dim=node_features
        # )

    def pass_messages(self, adj_torch, x):
        for dense in self.denses:
            x = F.dropout(x, self.dropout, training=self.training)
            x = adj_torch.matmul(x)
            x = dense(x)
        return x
    
    def forward(self, g, x, e, pe, adj_torch):

        degree = pe[:, 0:2]
        pos  = pe[:, 2:]

        # x = torch.cat((x, degree), dim=1)
        # x = self.pass_messages(adj_torch, x)
        # x = torch.cat((x, pos), dim=1)
        ## x = self.centrality_encoding(x, g)
        x = self.linear1_node(degree) # for genome
        ## x = F.dropout(x, self.dropout, training=self.training)

        e = self.linear1_edge(e)
        ## e = F.dropout(e, self.dropout, training=self.training)

        # 多层多头注意力机制
        for layer in self.attention_layers:
            node_outputs = []
            edge_outputs = []
            for att in layer:
                node_out, edge_out = att(g, x, e)
                node_outputs.append(node_out)
                edge_outputs.append(edge_out)

            x = torch.cat(node_outputs, dim=-1)
            e = torch.cat(edge_outputs, dim=-1)

            x = self.linear2_node(x)
            e = self.linear2_edge(e)
            
        scores = self.predictor(g, x, e)

        return scores

