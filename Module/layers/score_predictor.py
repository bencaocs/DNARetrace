import torch
import torch.nn as nn
import layers


class ScorePredictor(nn.Module):
    def __init__(self, in_features, hidden_edge_scores): 
        super().__init__()
        self.W1 = nn.Linear(3 * in_features, hidden_edge_scores) 
        self.W2 = nn.Linear(hidden_edge_scores, 1)
        
        # self.W = layers.NaiveFourierKANLayer(3 * in_features, 1)

    def apply_edges(self, edges):
        data = torch.cat((edges.src['x'], edges.dst['x'], edges.data['e']), dim=1)
        h = self.W1(data) 
        h = torch.relu(h)
        score = self.W2(h) 
        # score = self.W(data) 
        return {'score': score}

    def forward(self, graph, x, e):
        with graph.local_scope():
            graph.ndata['x'] = x
            graph.edata['e'] = e
            graph.apply_edges(self.apply_edges)
            
            return graph.edata['score']

