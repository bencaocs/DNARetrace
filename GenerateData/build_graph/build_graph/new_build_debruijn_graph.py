from imports import *
import pickle
import sys
import os
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqIO.FastaIO import FastaWriter
from functools import partial
sys.path.append("/root/data1/kISS/build")
import fm_index # 报错没事

class build_debruijn_graph:
    def __init__(self, k, reads, reference_sequence, save_graph_path, save_nodes_path, i, combine_fasta_path):
        self.k = k
        self.nodes_path = save_nodes_path
        self.i = i
        self.combine_fasta_path = combine_fasta_path
        self.save_graph_path = save_graph_path
        self.fmi = None
        self.reads, self.reference_sequence = self.analyze_data(reads, reference_sequence)
        self.build_fmi()
        self.nodes, self.edges = self.build_graph()
        self.dgl_graph = self.build_dgl_graph()
        self.save_graph()
    
    def build_fmi(self):
        combined_sequence = "".join(self.reference_sequence)
        seq_end = set()
        for a in range(199,len(combined_sequence),200):  #200 * 10000
            seq_end.add(a)
        record = SeqRecord(
                            Seq(combined_sequence),
                            id="CombinedSequence",  # FASTA 的序列标识符
                            description="This is a concatenated sequence from the list."  # 描述
                        )
        with open(self.combine_fasta_path , "w") as fasta_file:
            writer = FastaWriter(fasta_file, wrap=None)  # `wrap=None` 保证不换行
            writer.write_file([record])
        os.system(f'/root/data1/kISS/build/kISS suffix_sort {self.combine_fasta_path} -k 256 -t 4 --verbose')
        os.system(f'/root/data1/kISS/build/kISS fmindex_build {self.combine_fasta_path} -k 256 -t 4')
        self.fmi = fm_index.FMIndex_Uint32_KISS1(seq_end)
        self.fmi.load(self.combine_fasta_path + ".fmi")

    def save_nodes_to_pickle(self, nodes, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(nodes, f)

    def load_nodes_from_pickle(self, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
        
    def save_graph(self):
        dgl.save_graphs(self.save_graph_path, self.dgl_graph)

    def build_dgl_graph(self):
        DiGraph = nx.DiGraph() #有向图
        DiGraph.add_nodes_from((node, {
                # 'node': [attributes[0]], \
                'x': attributes[1],\
                }) 
                for node, attributes in self.nodes.items())
        
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
                            # 'edge': attributes[0],
                            'e': [attributes[1], attributes[2]],
                            'y': attributes[3]
                        }
                    )
                ]
            )
        
        # print("图的边信息:")
        # for edge in DiGraph.edges(data=True):
        #         print(f"边 {edge[0]} -> {edge[1]}: edge {edge[2]['edge']}, edge_abundance {edge[2]['edge_abundance']}, label {edge[2]['label']}")
        

        dgl_graph = dgl.from_networkx(DiGraph, node_attrs=['x'], edge_attrs=['e', 'y'])

        # 获取节点特征
        node_features = dgl_graph.ndata['x'].float()
        # 计算每个特征的均值和标准差
        n_mean_vals = torch.mean(node_features, dim=0)
        n_std_vals = torch.std(node_features, dim=0)

        # Z-score标准化
        n_normalized_feature = (node_features - n_mean_vals) / n_std_vals
        n_normalized_feature = torch.round(n_normalized_feature * 10000) / 10000  # 保留小数点后4位
        
        # 更新图的节点特征
        normalized_node_features = torch.stack([n_normalized_feature], dim=1)
        dgl_graph.ndata['x'] = normalized_node_features

        # 获取边特征
        edge_features = dgl_graph.edata['e'].float()

        # 将特征拆分为单独的张量
        e_feature1 = edge_features[:, 0]
        e_feature2 = edge_features[:, 1]

        # 分别计算每个特征的均值和标准差
        e_mean_vals1 = torch.mean(e_feature1, dim=0)
        e_std_vals1 = torch.std(e_feature1, dim=0)
        e_mean_vals2 = torch.mean(e_feature2, dim=0)
        e_std_vals2 = torch.std(e_feature2, dim=0)

        # 分别对每个特征进行 Z-score 标准化
        e_normalized_feature1 = (e_feature1 - e_mean_vals1) / e_std_vals1
        e_normalized_feature2 = (e_feature2 - e_mean_vals2) / e_std_vals2

        # 保留小数点后 4 位
        e_normalized_feature1 = torch.round(e_normalized_feature1 * 10000) / 10000
        e_normalized_feature2 = torch.round(e_normalized_feature2 * 10000) / 10000

        # 将标准化后的特征重新组合
        normalized_edge_features = torch.stack([e_normalized_feature1, e_normalized_feature2], dim=1)

        # 更新图的边特征
        dgl_graph.edata['e'] = normalized_edge_features
        
        return dgl_graph


    def analyze_data(self, reads, reference_sequence):
        reads = get_reads_from_fasta(reads)
        reference_sequence = get_reads_from_fasta(reference_sequence)
        return reads, reference_sequence
    
    def build_graph(self):
        nodes = dict()
        edges = dict()
        kmer_index_map = dict()
        deG = DeBruijnGraph(self.k)
        print('Adding seqs')
        deG.add_seqs(self.reads)
        kmers_abundance = deG.kmers # kmers is a dictionary of k-mers and their corresponding ubundance
        # print(f'Number of nodes: {len(deG.kmers)}')
        print('START GENERATING NODES...')
        for index, kmer in enumerate(kmers_abundance.keys()):
            nodes[index] = [kmer, kmers_abundance[kmer]]
            kmer_index_map[kmer] = index
        
        if self.i == 8 or self.i == 9:
            nodes_path = self.nodes_path + f"nodes_{self.i}.pkl"
            self.save_nodes_to_pickle(nodes, nodes_path)
        print(f'Number of nodes: {len(nodes)}')
        
        print('START GENERATING EDGES...')
        next_bases = {'1': 'A', '2': 'T', '3': 'C', '4': 'G'}
        for kmer, current_kmer_index in kmer_index_map.items():
            # print(f'Processing k-mer {kmer}...')

            suf_sub_kmer = kmer[1:]

            suf_kmer1 = suf_sub_kmer + next_bases['1']
            suf_kmer2 = suf_sub_kmer + next_bases['2']
            suf_kmer3 = suf_sub_kmer + next_bases['3']
            suf_kmer4 = suf_sub_kmer + next_bases['4']

            if suf_kmer1 in kmer_index_map:
                edge = kmer + next_bases['1']
                abundance = (kmers_abundance[kmer] + kmers_abundance[suf_kmer1]) / 2
                # label = self.label_for_edges(edge)
                occurrence_similairty = abs(kmers_abundance[suf_kmer1] - kmers_abundance[kmer])
                edges[(current_kmer_index, kmer_index_map[suf_kmer1])] = [edge, float(abundance), float(occurrence_similairty)]
            if suf_kmer2 in kmer_index_map:
                edge = kmer + next_bases['2']
                abundance = (kmers_abundance[kmer] + kmers_abundance[suf_kmer2]) / 2
                # label = self.label_for_edges(edge)
                occurrence_similairty = abs(kmers_abundance[suf_kmer2] - kmers_abundance[kmer])
                edges[(current_kmer_index, kmer_index_map[suf_kmer2])] = [edge, float(abundance), float(occurrence_similairty)]
            if suf_kmer3 in kmer_index_map:
                edge = kmer + next_bases['3']
                abundance = (kmers_abundance[kmer] + kmers_abundance[suf_kmer3]) / 2
                # label = self.label_for_edges(edge)
                occurrence_similairty = abs(kmers_abundance[suf_kmer3] - kmers_abundance[kmer])
                edges[(current_kmer_index, kmer_index_map[suf_kmer3])] = [edge, float(abundance), float(occurrence_similairty)]
            if suf_kmer4 in kmer_index_map:
                edge = kmer + next_bases['4']
                abundance = (kmers_abundance[kmer] + kmers_abundance[suf_kmer4]) / 2
                # label = self.label_for_edges(edge)
                occurrence_similairty = abs(kmers_abundance[suf_kmer4] - kmers_abundance[kmer])
                edges[(current_kmer_index, kmer_index_map[suf_kmer4])] = [edge, float(abundance), float(occurrence_similairty)]
        

        for edge_index, edge in edges.items():
            label = self.label_for_edges(edge[0])
            edges[edge_index].append(label)
        
        print(f'Number of edges: {len(edges)}')
        return nodes, edges
    
    def label_for_edges(self, edge):
        result = self.fmi.get_range(edge, 0)
        res = self.fmi.get_offsets(result[0],result[1])
        if len(res) > 0:
            return 1.0
        else:
            return 0.0

if __name__ == '__main__':
    k = 25
    for i in range(0, 10):
    # i = 8
        print(f"Processing data {i}...")
        train = '/root/data1/read/'
        reference = '/root/data1/reference/'
        grapn = '/root/data1/graph/'
        nodes_path = '/root/data1/graph/'

        fasta_file = f"{train}P10_data_{i}.fasta"
        reference_sequence = f"{reference}P10_reference_{i}.fasta"
        save_test_graph_path = f"{grapn}graph_kmer_{k}_{i}.bin"
        combine_fasta_path = f"{reference}P10_reference_{i}_combined.fasta"  
        deG = build_debruijn_graph(k, fasta_file, reference_sequence, save_test_graph_path, nodes_path, i, combine_fasta_path)