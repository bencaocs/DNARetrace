import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl.function as fn
import dgl
import layers

class GATLayer(nn.Module):
    def __init__(self, hidden_features, dropout, alpha, batch_norm, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = hidden_features
        self.out_features = hidden_features
        self.edge_features = hidden_features
        self.dropout = dropout
        self.alpha = alpha
        self.residual = True

        self.W_f = nn.Parameter(torch.zeros(size=(self.in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W_f.data, gain=1.414)
        self.We_f = nn.Parameter(torch.zeros(size=(self.edge_features, self.out_features)))
        nn.init.xavier_uniform_(self.We_f.data, gain=1.414)
        self.a_f = nn.Parameter(torch.zeros(size=(3 * self.out_features, 1)))
        nn.init.xavier_uniform_(self.a_f.data, gain=1.414)
        # self.W_f = layers.NaiveFourierKANLayer(self.in_features, self.out_features)
        # self.We_f = layers.NaiveFourierKANLayer(self.edge_features, self.out_features)
        # self.a_f = layers.NaiveFourierKANLayer(3*self.out_features, 1)

        self.W_r = nn.Parameter(torch.zeros(size=(self.in_features, self.out_features)))
        nn.init.xavier_uniform_(self.W_r.data, gain=1.414)
        self.We_r = nn.Parameter(torch.zeros(size=(self.edge_features, self.out_features)))
        nn.init.xavier_uniform_(self.We_r.data, gain=1.414)
        self.a_r = nn.Parameter(torch.zeros(size=(3 * self.out_features, 1)))
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        # self.W_r = layers.NaiveFourierKANLayer(self.in_features, self.out_features)
        # self.We_r = layers.NaiveFourierKANLayer(self.edge_features, self.out_features)
        # self.a_r = layers.NaiveFourierKANLayer(3*self.out_features, 1)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.B_1 = nn.Linear(self.in_features, self.out_features)
        self.B_2 = nn.Linear(self.in_features, self.out_features)
        self.B_3 = nn.Linear(self.in_features, self.out_features)
        
        if batch_norm:
            self.bn_h = nn.BatchNorm1d(self.out_features, track_running_stats=False)
            self.bn_e = nn.BatchNorm1d(self.out_features, track_running_stats=False)

    def forward(self, g, input_h, input_e):
        h_in = input_h.clone()
        e_in = input_e.clone()

        # g.ndata['B1h'] = self.B_1(input_h)
        # g.ndata['B2h'] = self.B_2(input_h)
        # g.edata['B3e'] = self.B_3(input_e)

        h_f = torch.mm(input_h, self.W_f)
        e_f = torch.mm(input_e, self.We_f)
        # h_f = self.W_f(input_h)
        # e_f = self.We_f(input_e)

        g.ndata['Wh'] = h_f
        g.edata['We'] = e_f
        
        g_r = dgl.reverse(g, copy_ndata=True, copy_edata=True)
        
        h_r = torch.mm(input_h, self.W_r)
        e_r = torch.mm(input_e, self.We_r)
        # h_r = self.W_r(input_h)
        # e_r = self.We_r(input_e)

        g_r.ndata['Wh'] = h_r
        g_r.edata['We'] = e_r

        src_nodes, dst_nodes = g.edges()
        edge_features = torch.cat([g.ndata['Wh'][src_nodes], g.ndata['Wh'][dst_nodes]], dim=1)
        edge_features = torch.cat([edge_features, g.edata['We']], dim=1)

        attention = self.leakyrelu(torch.matmul(edge_features, self.a_f).squeeze(1))
        # attention = self.leakyrelu(self.a_f(edge_features).squeeze(1))

        # 直接对注意力分数进行 softmax 处理，得到归一化的注意力得分
        g.edata['alpha'] = dgl.ops.edge_softmax(g, attention)

        attention = g.edata['alpha']

        # 使用 dropout 防止过拟合
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 注意力加权特征求和，更新节点表示
        g.edata['aWh'] = attention.unsqueeze(-1) * g.edata['We']  # 修正边特征
        g.update_all(fn.copy_edge('aWh', 'm'), fn.sum('m', 'h_sum'))

        ##########################反向图#################################
        src_nodes_r, dst_nodes_r = g_r.edges()
        edge_features_r = torch.cat([g_r.ndata['Wh'][src_nodes_r], g_r.ndata['Wh'][dst_nodes_r]], dim=1)
        edge_features_r = torch.cat([edge_features_r, g_r.edata['We']], dim=1)

        attention_r = self.leakyrelu(torch.matmul(edge_features_r, self.a_r).squeeze(1))
        # attention_r = self.leakyrelu(self.a_r(edge_features).squeeze(1))

        g_r.edata['alpha'] = dgl.ops.edge_softmax(g_r, attention_r)

        attention_r = g_r.edata['alpha']

        attention_r = F.dropout(attention_r, self.dropout, training=self.training)

        g_r.edata['aWh'] = attention_r.unsqueeze(-1) * g_r.edata['We']  # 修正边特征
        g_r.update_all(fn.copy_edge('aWh', 'm'), fn.sum('m', 'h_sum'))

        output_h = g.ndata['h_sum'] + g_r.ndata['h_sum'] + g.ndata['Wh']
        g.ndata['h_update'] = output_h
        output_h = self.bn_h(output_h)
        output_h = torch.relu(output_h)
        if self.residual:
            output_h = output_h + h_in

        g.apply_edges(fn.u_add_v('h_update', 'h_update', 'h_new'))
        e_ji = g.edata['h_new'] + g.edata['We']
        e_ji = self.bn_e(e_ji)
        e_ji = torch.relu(e_ji)
        if self.residual:
            e_ji = e_ji + e_in
        output_e = e_ji

        return output_h, output_e

