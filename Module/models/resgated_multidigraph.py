import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# from torch_geometric.nn.conv import MessagePassing
import dgl
import dgl.function as fn


class resgated_multidigraph(nn.Module):
    def __init__(
        self,
        num_layers: int = 1,
        node_features: int = 2,
        edge_features: int = 1,
        hidden_features: int = 64,
    ):
        super().__init__()

        self.W11 = nn.Linear(node_features, hidden_features, bias=True)
        self.W12 = nn.Linear(hidden_features, hidden_features, bias=True)
        self.W21 = nn.Linear(edge_features, hidden_features, bias=True)
        self.W22 = nn.Linear(hidden_features, hidden_features, bias=True)

        self.gate = LayeredGatedGCN(num_layers=num_layers, hidden_features=hidden_features)
        self.ln1 = nn.LayerNorm(hidden_features)
        self.ln2 = nn.LayerNorm(hidden_features)

        self.scorer1 = nn.Linear(3 * hidden_features, hidden_features, bias=True)
        self.scorer2 = nn.Linear(hidden_features, out_features=1, bias=True)
        self.scorer = nn.Linear(3 * hidden_features, out_features=1, bias=True)

        self.threshold = 0.7

    def apply_edges(self, edges):
        data = torch.cat((edges.src['x'], edges.dst['x'], edges.data['e']), dim=1)
        
        h = self.scorer1(data) 
        h = torch.relu(h)
        score = self.scorer2(h) 
    
        return {'score': score}
    
    def forward(self, graph, x, e) -> Tensor:
        x = self.W12(self.ln1(torch.relu(self.W11(x))))
        e = self.W22(self.ln2(torch.relu(self.W21(e))))

        x, e = self.gate.forward(graph, x=x, e=e)

        with graph.local_scope():
            graph.ndata['x'] = x
            graph.edata['e'] = e
            graph.apply_edges(self.apply_edges)

            return graph.edata['score']


class LayeredGatedGCN(nn.Module):
    def __init__(self, num_layers: int, hidden_features: int, gate_norm: str = "feature"):
        super().__init__()
        self.gnn = nn.ModuleList(
            GatedGCN(hidden_features=hidden_features, gate_norm=gate_norm) for _ in range(num_layers)
        )

    def forward(self, graph, x, e):
        for gnn_layer in self.gnn:
            x, e = gnn_layer(g=graph, h=x, e=e)
        return x, e

class GatedGCN(nn.Module):
    def __init__(self, hidden_features: int, gate_norm: str, layer_norm: bool = True):
        super().__init__()
        self.gate_norm = gate_norm
        self.layer_norm = layer_norm

        self.A1 = nn.Linear(hidden_features, hidden_features)
        self.A2 = nn.Linear(hidden_features, hidden_features)
        self.A3 = nn.Linear(hidden_features, hidden_features)

        self.B1 = nn.Linear(hidden_features, hidden_features)
        self.B2 = nn.Linear(hidden_features, hidden_features)
        self.B3 = nn.Linear(hidden_features, hidden_features)

        if self.layer_norm:
            self.ln_h = nn.LayerNorm(hidden_features)
            self.ln_e = nn.LayerNorm(hidden_features)

    def forward(self, g, h, e):
        with g.local_scope():  # DGL推荐使用local_scope()避免在原图上修改
            A1h = self.A1(h)
            A2h = self.A2(h)
            A3h = self.A3(h)

            B1h = self.B1(e)
            B2h = self.B2(h)
            B3h = self.B3(h)

            # 获取图的边的源和目标节点
            src, dst = g.edges()

            # 计算边的特征
            e_fw = B1h + B2h[src] + B3h[dst]
            e_bw = B1h + B2h[dst] + B3h[src]

            e_fw = F.relu(e_fw)
            e_bw = F.relu(e_bw)

            if self.layer_norm:
                e_fw = self.ln_e(e_fw)
                e_bw = self.ln_e(e_bw)

            # 残差连接
            e_fw = e + e_fw
            e_bw = e + e_bw

            sigmoid_fw = torch.sigmoid(e_fw)
            sigmoid_bw = torch.sigmoid(e_bw)

            g.edata['sigma_fw'] = sigmoid_fw
            g.edata['sigma_bw'] = sigmoid_bw

            g.ndata['A2h'] = A2h
            g.ndata['A3h'] = A3h

            # 消息传递
            g.update_all(
                message_func=self.message_fw,
                reduce_func=fn.sum(msg='msg', out='h_fw')
            )

            g.update_all(
                message_func=self.message_bw,
                reduce_func=fn.sum(msg='msg', out='h_bw')
            )

            h_fw = g.ndata.pop('h_fw')
            h_bw = g.ndata.pop('h_bw')

            h_new = A1h + h_fw + h_bw
            h_new = F.relu(h_new)
            if self.layer_norm:
                h_new = self.ln_h(h_new)
            h = h + h_new

            return h, e_fw

    def message_fw(self, edges):
        msg = edges.src['A2h'] * edges.data['sigma_fw']
        if self.gate_norm == "feature":
            msg = msg / (torch.sum(edges.data['sigma_fw'], dim=1).unsqueeze(dim=1) + 1e-6)
        return {'msg': msg}

    def message_bw(self, edges):
        msg = edges.src['A3h'] * edges.data['sigma_bw']
        if self.gate_norm == "feature":
            msg = msg / (torch.sum(edges.data['sigma_bw'], dim=1).unsqueeze(dim=1) + 1e-6)
        return {'msg': msg}

# class GatedGCN(MessagePassing):
#     def __init__(self, hidden_features: int, gate_norm: str, layer_norm: bool = True):
#         super().__init__()
#         self.gate_norm = gate_norm
#         self.layer_norm = layer_norm

#         self.A1 = nn.Linear(hidden_features, hidden_features)
#         self.A2 = nn.Linear(hidden_features, hidden_features)
#         self.A3 = nn.Linear(hidden_features, hidden_features)

#         self.B1 = nn.Linear(hidden_features, hidden_features)
#         self.B2 = nn.Linear(hidden_features, hidden_features)
#         self.B3 = nn.Linear(hidden_features, hidden_features)

#         if self.layer_norm:
#             self.ln_h = nn.LayerNorm(hidden_features)
#             self.ln_e = nn.LayerNorm(hidden_features)

#     def forward(self, g, h, e):
#         A1h = self.A1(h)
#         A2h = self.A2(h)
#         A3h = self.A3(h)

#         B1h = self.B1(e)
#         B2h = self.B2(h)
#         B3h = self.B3(h)

#         src, dst = edge_index
#         bw_edge_index = torch.vstack((dst, src))

#         e_fw = B1h + B2h[src] + B3h[dst]
#         e_bw = B1h + B2h[dst] + B3h[src]

#         e_fw = F.relu(e_fw)
#         e_bw = F.relu(e_bw)

#         if self.layer_norm:
#             e_fw = self.ln_e(e_fw)
#             e_bw = self.ln_e(e_bw)

#         # residual connection
#         e_fw = edge_attr + e_fw
#         e_bw = edge_attr + e_bw

#         sigmoid_fw = torch.sigmoid(e_fw)
#         sigmoid_bw = torch.sigmoid(e_bw)

#         h_fw = self.propagate(edge_index=edge_index, x=A2h, sigma=sigmoid_fw)
#         h_bw = self.propagate(edge_index=bw_edge_index, x=A3h, sigma=sigmoid_bw)

#         h_new = A1h + h_fw + h_bw
#         h_new = F.relu(h_new)
#         if self.layer_norm:
#             h_new = self.ln_h(h_new)
#         h = h + h_new

#         return h, e_fw

#     def message(self, x_j, sigma) -> Tensor:
#         # in pyg (j->i) represents the flow from source to target and (i->j) the reverse
#         # generally, i is the node that accumulates information and {j} its neighbors
#         message = x_j * sigma
#         if self.gate_norm == "feature":
#             message = message / (torch.sum(sigma, dim=1).unsqueeze(dim=1) + 1e-6)
#         return message
