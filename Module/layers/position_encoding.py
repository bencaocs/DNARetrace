import torch
from torch import nn


class CentralityEncoding(nn.Module):
    def __init__(self, max_in_degree: int, max_out_degree: int, node_dim: int):
        """
        :param max_in_degree: max in degree of nodes
        :param max_out_degree: max out degree of nodes
        :param node_dim: hidden dimensions of node features
        """
        super().__init__()
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.node_dim = node_dim
        self.z_in = nn.Parameter(torch.randn((max_in_degree, node_dim)))
        self.z_out = nn.Parameter(torch.randn((max_out_degree, node_dim)))

    def forward(self, x, g):
        """
        :param x: node feature matrix
        :param edge_index: edge_index of graph (adjacency list)
        :return: torch.Tensor, node embeddings after Centrality encoding
        """

        in_degree = self.decrease_to_max_value(g.in_degrees().long(),
                                                self.max_in_degree - 1) # 每个节点的入度
        out_degree = self.decrease_to_max_value(g.out_degrees().long(),
                                                self.max_out_degree - 1)

        x += self.z_in[in_degree] + self.z_out[out_degree] # 将每个节点度的数值作为索引，挑选z_in或z_out的每行，形成每个节点的嵌入

        return x

    def decrease_to_max_value(self, x, max_value):
        "限制节点度的最大值"
        x[x > max_value] = max_value

        return x