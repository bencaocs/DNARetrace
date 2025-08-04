from imports import *
import pickle

class build_debruijn_graph:
    def __init__(self, k, reads, reference_sequence, save_graph_path, save_nodes_path, i):
        self.k = k
        self.nodes_path = save_nodes_path
        self.i = i
        self.reads, self.reference_sequence = self.analyze_data(reads, reference_sequence)
        self.save_graph_path = save_graph_path
        self.nodes, self.edges = self.build_graph()
        self.dgl_graph = self.build_dgl_graph()
        self.save_graph()

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
        # node_features = dgl_graph.ndata['x']
        #最小-最大归一化
        # n_min_vals, _  = torch.min(node_features, dim=0)
        # n_max_vals, _  = torch.max(node_features, dim=0)

        # n_normalized_feature = (node_features - n_min_vals) / (n_max_vals - n_min_vals)
        # n_normalized_feature = torch.round(n_normalized_feature * 10000) / 10000

        # normalized_node_features = torch.stack([n_normalized_feature], dim=1)
        # dgl_graph.ndata['x'] = normalized_node_features
        #   
        # # 获取边特征
        # edge_features = dgl_graph.edata['e']

        # # 将特征拆分为单独的张量
        # e_feature1 = edge_features[:, 0]
        # e_feature2 = edge_features[:, 1]

        # # 分别计算每个特征的最小值和最大值
        # e_min_vals1, _  = torch.min(e_feature1, dim=0)
        # e_max_vals1, _  = torch.max(e_feature1, dim=0)
        # e_min_vals2, _  = torch.min(e_feature2, dim=0)
        # e_max_vals2, _  = torch.max(e_feature2, dim=0)

        # # 分别对每个特征进行最小-最大归一化
        # e_normalized_feature1 = (e_feature1 - e_min_vals1) / (e_max_vals1 - e_min_vals1)
        # e_normalized_feature2 = (e_feature2 - e_min_vals2) / (e_max_vals2 - e_min_vals2)

        # e_normalized_feature1 = torch.round(e_normalized_feature1 * 10000) / 10000
        # e_normalized_feature2 = torch.round(e_normalized_feature2 * 10000) / 10000

        # # 将归一化后的特征重新组合
        # normalized_edge_features = torch.stack([e_normalized_feature1, e_normalized_feature2], dim=1)

        # dgl_graph.edata['e'] = normalized_edge_features
        
        #最小-最大归一化##############
        
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
    
    def label_for_edges(self, edge):
        for seq in self.reference_sequence:
            if edge in seq:
                return 1
        return 0

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
            print(f'Processing k-mer {kmer}...')
            # pre_sub_kmer = kmer[:-1]
            suf_sub_kmer = kmer[1:]

            # pre_kmer1 = next_bases['1'] + pre_sub_kmer
            # pre_kmer2 = next_bases['2'] + pre_sub_kmer
            # pre_kmer3 = next_bases['3'] + pre_sub_kmer
            # pre_kmer4 = next_bases['4'] + pre_sub_kmer

            suf_kmer1 = suf_sub_kmer + next_bases['1']
            suf_kmer2 = suf_sub_kmer + next_bases['2']
            suf_kmer3 = suf_sub_kmer + next_bases['3']
            suf_kmer4 = suf_sub_kmer + next_bases['4']

            # if pre_kmer1 in kmer_index_map:
            #     edge = next_bases['1'] + kmer
            #     abundance = (kmers_abundance[pre_kmer1] + kmers_abundance[kmer]) / 2
            #     label = self.label_for_edges(edge)
            #     occurrence_similairty = abs(kmers_abundance[pre_kmer1] - kmers_abundance[kmer])
            #     edges[(kmer_index_map[pre_kmer1], current_kmer_index)] = [edge, float(abundance), float(occurrence_similairty), float(label)]
            # if pre_kmer2 in kmer_index_map:
            #     edge = next_bases['2'] + kmer
            #     abundance = (kmers_abundance[pre_kmer2] + kmers_abundance[kmer]) / 2
            #     label = self.label_for_edges(edge)
            #     occurrence_similairty = abs(kmers_abundance[pre_kmer2] - kmers_abundance[kmer])
            #     edges[(kmer_index_map[pre_kmer2], current_kmer_index)] = [edge, float(abundance), float(occurrence_similairty), float(label)]
            # if pre_kmer3 in kmer_index_map:
            #     edge = next_bases['3'] + kmer
            #     abundance = (kmers_abundance[pre_kmer3] + kmers_abundance[kmer]) / 2
            #     label = self.label_for_edges(edge)
            #     occurrence_similairty = abs(kmers_abundance[pre_kmer3] - kmers_abundance[kmer])
            #     edges[(kmer_index_map[pre_kmer3], current_kmer_index)] = [edge, float(abundance), float(occurrence_similairty), float(label)]
            # if pre_kmer4 in kmer_index_map:
            #     edge = next_bases['4'] + kmer
            #     abundance = (kmers_abundance[pre_kmer4] + kmers_abundance[kmer]) / 2
            #     label = self.label_for_edges(edge)
            #     occurrence_similairty = abs(kmers_abundance[pre_kmer4] - kmers_abundance[kmer])
            #     edges[(kmer_index_map[pre_kmer4], current_kmer_index)] = [edge, float(abundance), float(occurrence_similairty), float(label)]

            if suf_kmer1 in kmer_index_map:
                edge = kmer + next_bases['1']
                abundance = (kmers_abundance[kmer] + kmers_abundance[suf_kmer1]) / 2
                label = self.label_for_edges(edge)
                occurrence_similairty = abs(kmers_abundance[suf_kmer1] - kmers_abundance[kmer])
                edges[(current_kmer_index, kmer_index_map[suf_kmer1])] = [edge, float(abundance), float(occurrence_similairty), float(label)]
            if suf_kmer2 in kmer_index_map:
                edge = kmer + next_bases['2']
                abundance = (kmers_abundance[kmer] + kmers_abundance[suf_kmer2]) / 2
                label = self.label_for_edges(edge)
                occurrence_similairty = abs(kmers_abundance[suf_kmer2] - kmers_abundance[kmer])
                edges[(current_kmer_index, kmer_index_map[suf_kmer2])] = [edge, float(abundance), float(occurrence_similairty), float(label)]
            if suf_kmer3 in kmer_index_map:
                edge = kmer + next_bases['3']
                abundance = (kmers_abundance[kmer] + kmers_abundance[suf_kmer3]) / 2
                label = self.label_for_edges(edge)
                occurrence_similairty = abs(kmers_abundance[suf_kmer3] - kmers_abundance[kmer])
                edges[(current_kmer_index, kmer_index_map[suf_kmer3])] = [edge, float(abundance), float(occurrence_similairty), float(label)]
            if suf_kmer4 in kmer_index_map:
                edge = kmer + next_bases['4']
                abundance = (kmers_abundance[kmer] + kmers_abundance[suf_kmer4]) / 2
                label = self.label_for_edges(edge)
                occurrence_similairty = abs(kmers_abundance[suf_kmer4] - kmers_abundance[kmer])
                edges[(current_kmer_index, kmer_index_map[suf_kmer4])] = [edge, float(abundance), float(occurrence_similairty), float(label)]
        print(f'Number of edges: {len(edges)}')
        return nodes, edges


if __name__ == '__main__':
    k = 26
    for i in range(5, 10):
    # i = 8
        train = 'id20/read/'
        reference = 'id20/reference/'
        grapn = 'id20/graph/'
        nodes_path = 'id20/graph/'

        fasta_file = f"{train}id20_read_{i}.fasta"
        reference_sequence = f"{reference}id20_reference_{i}.fasta"
        save_test_graph_path = f"{grapn}graph_kmer_{k}_{i}.bin"
        deG = build_debruijn_graph(k, fasta_file, reference_sequence, save_test_graph_path, nodes_path, i)