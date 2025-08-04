from imports import *
import pickle
import sys
import os
from Bio import SeqIO
from Bio.Seq import Seq

from multiprocessing import Pool, Manager
sys.path.append("/root/data1/kISS/build")
import fm_index # 报错没事

class build_debruijn_graph:
    def __init__(self, reference_path, gfa_path, save_graph_path, save_node_path, save_edge_path, k):
        self.reference_path = reference_path
        self.gfa_path = gfa_path
        self.save_graph_path = save_graph_path
        self.save_node_path = save_node_path
        self.save_edge_path = save_edge_path
        self.k = k
        self.fmi = None
        self.reference, self.ref_len = self.load_reference()
        self.build_fmi()
        self.nodes, self.edges = self.build_graph()
        self.dgl_graph = self.build_dgl_graph()
        self.save_graph()
    
    def build_fmi(self):
        seq_end = set()
        # for a in range(199,200000,200):  #200 * 1000
        seq_end.add(self.ref_len + 1)
        # concatenate_fasta_to_single_line(f'/home/lzz/Projects/KANGNN_for_Assembly/build_graph/0days/reference/0days_reference_{i}.fasta','0days_reference.fa')
        os.system(f'/root/data1/kISS/build/kISS suffix_sort {self.reference_path} -k 256 -t 4 --verbose')
        os.system(f'/root/data1/kISS/build/kISS fmindex_build {self.reference_path} -k 256 -t 4')
        self.fmi = fm_index.FMIndex_Uint32_KISS1(seq_end)
        self.fmi.load(self.reference_path + ".fmi")


    def load_reference(self):
        sequences = []
        for record in SeqIO.parse(self.reference_path, "fasta"):
            sequences.append(str(record.seq))  # 将序列作为字符串存入列表
        
        return sequences, len(sequences[0])
    
    def save_to_pickle(self, nodes, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(nodes, f)

    def save_graph(self):
        dgl.save_graphs(self.save_graph_path, self.dgl_graph)
    
    def build_graph(self):
        nodes = dict()
        edges = dict()
        print('START GENERATING NODES...')
        with open(self.gfa_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("S"):
                    parts = line.split("\t")
                    node_index = float(parts[1])
                    node_seq = str(parts[2])
                    node_len = float(parts[3].split(":")[-1])
                    node_kc = float(parts[4].split(":")[-1])
                    node_km = float(parts[5].split(":")[-1])
                    
                    nodes[node_index] = [node_seq, node_len, node_km, node_kc]
        
        with open(self.gfa_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("L"):
                    parts = line.split("\t")
                    pre_node_index = float(parts[1])
                    suf_node_index = float(parts[3])
                    pre_node_singal = str(parts[2])
                    suf_node_singal = str(parts[4])
                    over_lap_len = float(parts[5][:-1])
                    edge_len = float(nodes[pre_node_index][1]+nodes[suf_node_index][1]-over_lap_len)
                    edge_km = float((nodes[pre_node_index][3]+nodes[suf_node_index][3])/(edge_len-self.k+1))
                    edge_occurrence = float(abs(nodes[pre_node_index][2]-nodes[suf_node_index][2]))
                    
                    if pre_node_singal == "+":
                        pre_node = nodes[pre_node_index][0]
                    else:
                        pre_node_tmp = nodes[pre_node_index][0]
                        pre_node = str(Seq(pre_node_tmp).reverse_complement())
                    if suf_node_singal == "+":
                        suf_node = nodes[suf_node_index][0]
                    else:
                        suf_node_tmp = nodes[suf_node_index][0]
                        suf_node = str(Seq(suf_node_tmp).reverse_complement())

                    edge = pre_node + suf_node[int(over_lap_len):]
                    label = self.label_for_edges(edge)

                    edges[(pre_node_index, suf_node_index)] = [edge_len, edge_km, edge_occurrence, label, pre_node_singal, suf_node_singal, over_lap_len]
        

        # print('START GENERATING EDGES...')
        # edges_label = self.parallel_edge_processing(edge_for_label, self.reference)
        # print(type(edges_label), edges_label)
        
        # for edge in edges:
        #     edges[edge].append(edges_label[edge])
        
        self.save_to_pickle(nodes, self.save_node_path)
        self.save_to_pickle(edges, self.save_edge_path)

        return nodes, edges
    
    def label_for_edges(self, edge):
        result = self.fmi.get_range(edge,0) # edge是一条序列 ATACGAGAGGAGGAGGG
        res = self.fmi.get_offsets(result[0],result[1])
        # rc_result = self.fmi.get_range(DNA_rev_complement(edge),0)
        # rc_res = self.fmi.get_offsets(rc_result[0],rc_result[1])
        if len(res) > 0:
            return 1.0
        else:
            return 0.0

    def build_dgl_graph(self):
        DiGraph = nx.DiGraph() #有向图
        # DiGraph.add_nodes_from((node, {'l': attributes[1], 'a': attributes[2],}) for node, attributes in self.nodes.items()) # l表示节点长度，a表示节点覆盖度
        for node, attributes in self.nodes.items():
            DiGraph.add_nodes_from(
                [   (
                        node,
                        {
                            'x': [attributes[1], attributes[2]] # 节点长度和覆盖度作为节点特征
                        }
                    )
                ]
            )
        
        # print("图的节点信息:")
        # for node, attributes in DiGraph.nodes(data=True):
        #     print(f"节点 {node}: {attributes}")

        for edge, attributes in self.edges.items():
            DiGraph.add_edges_from( 
                [
                    (
                        edge[0],
                        edge[1],
                        {
                            'e': [attributes[0], attributes[1], attributes[2]], # 边长度，边覆盖度，边的一致性作为边特征
                            'y': attributes[3]
                        }
                    )
                ]
            )
        
        print("图的边信息:")
        for edge in DiGraph.edges(data=True):
                print(f"边 {edge[0]} -> {edge[1]}: edge {edge[2]['e']}, label {edge[2]['y']}")
        

        dgl_graph = dgl.from_networkx(DiGraph, node_attrs=['x'], edge_attrs=['e', 'y'])
        
        # 获取节点特征
        node_features = dgl_graph.ndata['x'].float()
            
        # 将特征拆分为单独的张量
        n_feature1 = node_features[:, 0]
        n_feature2 = node_features[:, 1]
        
        # 分别计算每个特征的均值和标准差
        n_mean_vals1 = torch.mean(n_feature1, dim=0)
        n_std_vals1 = torch.std(n_feature1, dim=0)
        n_mean_vals2 = torch.mean(n_feature2, dim=0)
        n_std_vals2 = torch.std(n_feature2, dim=0)

        # 分别对每个特征进行 Z-score 标准化
        n_normalized_feature1 = (n_feature1 - n_mean_vals1) / n_std_vals1
        n_normalized_feature2 = (n_feature2 - n_mean_vals2) / n_std_vals2 
        
        # 保留小数点后 4 位
        n_normalized_feature1 = torch.round(n_normalized_feature1 * 10000) / 10000
        n_normalized_feature2 = torch.round(n_normalized_feature2 * 10000) / 10000

        # 将标准化后的特征重新组合
        normalized_node_features = torch.stack([n_normalized_feature1, n_normalized_feature2], dim=1)
        
        # 更新图的节点特征
        dgl_graph.ndata['x'] = normalized_node_features

        #################### 边特征 ###############

        # 获取边特征
        edge_features = dgl_graph.edata['e'].float()

        # 将特征拆分为单独的张量
        e_feature1 = edge_features[:, 0]
        e_feature2 = edge_features[:, 1]
        e_feature3 = edge_features[:, 2]

        # 分别计算每个特征的均值和标准差
        e_mean_vals1 = torch.mean(e_feature1, dim=0)
        e_std_vals1 = torch.std(e_feature1, dim=0)
        e_mean_vals2 = torch.mean(e_feature2, dim=0)
        e_std_vals2 = torch.std(e_feature2, dim=0)
        e_mean_vals3 = torch.mean(e_feature3, dim=0)
        e_std_vals3 = torch.std(e_feature3, dim=0)

        # 分别对每个特征进行 Z-score 标准化
        e_normalized_feature1 = (e_feature1 - e_mean_vals1) / e_std_vals1
        e_normalized_feature2 = (e_feature2 - e_mean_vals2) / e_std_vals2
        e_normalized_feature3 = (e_feature3 - e_mean_vals3) / e_std_vals3

        # 保留小数点后 4 位
        e_normalized_feature1 = torch.round(e_normalized_feature1 * 10000) / 10000
        e_normalized_feature2 = torch.round(e_normalized_feature2 * 10000) / 10000
        e_normalized_feature3 = torch.round(e_normalized_feature3 * 10000) / 10000

        # 将标准化后的特征重新组合
        normalized_edge_features = torch.stack([e_normalized_feature1, e_normalized_feature2, e_normalized_feature3], dim=1)

        # 更新图的边特征
        dgl_graph.edata['e'] = normalized_edge_features
        
        return dgl_graph
    
if __name__ == '__main__':
    for i in [17]: # , 8, 9, 10, 11, 12
        k = 501
        reference_path = f"/root/data1/genome/chr{i}.fasta" 
        gfa_path = f"/root/data1/chr{i}_0001.unitigs.gfa"
        save_graph_path = f"graph/graph_kmer_{k}_chr{i}.bin"
        save_node_path = f"graph/node_{k}_chr{i}.pkl"
        save_edge_path = f"graph/edge_{k}_chr{i}.pkl"
        
        deG = build_debruijn_graph(reference_path, gfa_path, save_graph_path, save_node_path, save_edge_path, k)