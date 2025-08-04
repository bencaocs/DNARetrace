from imports import *
import pickle
import sys
sys.path.append("/root/data1/kISS/build")
import subprocess
import argparse
import fm_index # 报错没事

class build_debruijn_graph:
    def __init__(self, k, reads, reference_sequence, save_graph_path, save_nodes_path, i, h):
        self.k = k
        self.nodes_path = save_nodes_path
        self.i = i
        self.h = h
        self.save_graph_path = save_graph_path
        self.fmi = None
        self.reads, self.reference_sequence = self.analyze_data(reads, reference_sequence)
        self.build_fmi(reference_sequence)
        self.nodes, self.edges = self.build_graph()
        self.dgl_graph = self.build_dgl_graph()
        self.save_graph()
    
    def build_fmi(self, reference_sequence):
        seq_end = set()
        seq_end.add(len(self.reference_sequence[0])+1)
        print(f"Building fmi for {len(self.reference_sequence[0])}")
        subprocess.run(f'/root/data1/kISS/build/kISS suffix_sort {reference_sequence} -k 256 -t 4 --verbose', shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(f'/root/data1/kISS/build/kISS fmindex_build {reference_sequence} -k 256 -t 4', shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.fmi = fm_index.FMIndex_Uint32_KISS1(seq_end)
        self.fmi.load(reference_sequence + ".fmi")

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
        
        kmers_abundance = {kmer: abundance for kmer, abundance in deG.kmers.items() if abundance >= self.h}
        print('START GENERATING NODES...')
        
        for index, (kmer, abundance) in enumerate(kmers_abundance.items()):
            nodes[index] = [kmer, abundance]
            kmer_index_map[kmer] = index
        
        if self.i == 19:
            nodes_path = self.nodes_path + f"nodes_{self.i}.pkl"
            self.save_nodes_to_pickle(nodes, nodes_path)
        print(f'Number of nodes: {len(nodes)}')
        
        print('START GENERATING EDGES...')
        next_bases = ['A', 'T', 'C', 'G']
        
        for kmer, current_kmer_index in kmer_index_map.items():
            suf_sub_kmer = kmer[1:]
            for next_base in next_bases:
                suf_kmer = suf_sub_kmer + next_base
                if suf_kmer in kmer_index_map:
                    edge = kmer + next_base
                    abundance = (kmers_abundance[kmer] + kmers_abundance[suf_kmer]) / 2
                    occurrence_similarity = abs(kmers_abundance[suf_kmer] - kmers_abundance[kmer])
                    label = self.label_for_edges(edge)
                    edges[current_kmer_index, kmer_index_map[suf_kmer]] = [edge, abundance, occurrence_similarity, label]   

        print(f'Number of edges: {len(edges)}')
        return nodes, edges
    
    def label_for_edges(self, edge):
        result = self.fmi.get_range(edge, 0)
        res = self.fmi.get_offsets(result[0],result[1])
        if len(res) > 0:
            return 1.0
        else:
            return 0.0

def main():
    parser = argparse.ArgumentParser(description="Process De Bruijn Graph tasks.")
    parser.add_argument('--i', type=int, required=True, help="The index of the data to process (e.g., 18, 19, etc.)")
    args = parser.parse_args()

    i = args.i
    print(f"Processing data {i}...")

    k = 251
    h = 1
    read = '/root/data1/read/'
    reference = '/root/data1/genome/'
    graph = '/root/data1/graph/'
    nodes_path = '/root/data1/graph/'

    fasta_file = f"{read}chr{i}_0001.fa"
    reference_sequence = f"{reference}chr{i}.fasta"
    save_test_graph_path = f"{graph}graph_kmer_{k}_{i}.bin"

    build_debruijn_graph(k, fasta_file, reference_sequence, save_test_graph_path, nodes_path, i, h)

if __name__ == '__main__':
    main()