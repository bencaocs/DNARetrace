import pickle
import networkx as nx
import dgl
from Bio import SeqIO
from Bio.Seq import Seq
from collections import defaultdict
import numpy as np
import Levenshtein
import matplotlib.pyplot as plt
import re
import pickle
import random
import sys

class AssemblyGraph:
    def __init__(self, nodes_filepath, DGL_filepath, reference_filepath, edges_filepath, kmer, contigs_filepath):
        self.nodes_filepath = nodes_filepath
        self.DGL_filepath = DGL_filepath
        self.kmer = kmer
        self.reference_filepath = reference_filepath
        self.edges = edges_filepath
        self.contigs_filepath = contigs_filepath
        self.reference_len, self.node2edge, self.edge2seq, self.edge2seq_2 = self.readile()
        self.start_nodes, self.nx_graph = self.get_start_node()
        self.assembly_graph(self.reference_len)
        # self.DGL2nx(self.DGL_filepath)
    
    def readile(self):
        edge2seq = {}
        edge2seq_2 = {}
        with open(self.reference_filepath, "r") as fasta_file:
            # 使用 SeqIO 读取 fasta 文件
            record = next(SeqIO.parse(fasta_file, "fasta"))
        reference_len = len(record.seq)
        print(f"Reference length: {reference_len}")

        with open(self.nodes_filepath, 'rb') as file:  # 使用二进制模式读取
            node2edge = pickle.load(file)
        # print(f"Node to edge map: {node2edge}")
        
        for record in SeqIO.parse(self.edges, "fasta"):
            edge2seq[record.id] = str(record.seq)  # 使用ID作为key，序列内容作为value
            match = re.match(r"-?(\d+\.\d+)_-?(\d+\.\d+)", record.id)
            if match:
                # 提取的两个数字
                num_list = sorted([str(match.group(1)), str(match.group(2))])
                num1, num2 = num_list  # 排序后的两个数字
                edge2seq_2[(num1, num2)] = str(record.seq)  # 使用ID作为key，序列内容作为value


        return reference_len, node2edge, edge2seq, edge2seq_2  # 返回序列的长度
    
    def get_start_node(self):
        graphs, _ = dgl.load_graphs(self.DGL_filepath)
        dgl_graph = graphs[0]

        # 获取DGL图中可用的节点和边的属性
        node_attrs = list(dgl_graph.ndata.keys())  # 获取所有节点属性的名称
        edge_attrs = list(dgl_graph.edata.keys())  # 获取所有边属性的名称
        print(f"Node attributes: {node_attrs}")
        print(f"Edge attributes: {edge_attrs}")
        # 将DGL图转换为NetworkX图
        nx_graph = dgl.to_networkx(dgl_graph, node_attrs=['node_id'], edge_attrs=['score', 'cov']) #'e':[cov, edge_len]
        
        # edges_with_cov = {(u, v): attr['cov'].item() for u, v, attr in nx_graph.edges(data=True)}
        # sorted_edges = sorted(edges_with_cov.items(), key=lambda item: item[1], reverse=True)
        # start_nodes = [u for (u, v), cov in sorted_edges[:10]]
        
        start_nodes = [node for node, in_deg in nx_graph.in_degree() if in_deg == 0][:4000]
        # start_nodes_log = [nx_graph.nodes[node]['node_id'] for node, in_deg in nx_graph.in_degree() if in_deg == 0][0:50]
        # start_nodes = list()
        # id = ['tensor(-10)', 'tensor(-100005)', 'tensor(-100011)', 'tensor(-100029)', 'tensor(-100094)', 'tensor(-100116)', 'tensor(-100156)', 'tensor(-100160)', 'tensor(-100165)', 'tensor(-100198)', 'tensor(-100236)', 'tensor(-100277)', 'tensor(-100281)', 'tensor(-100326)', 'tensor(-100329)', 'tensor(-100339)', 'tensor(-100393)', 'tensor(-100407)', 'tensor(-100425)', 'tensor(-100432)', 'tensor(-100449)', 'tensor(-100466)', 'tensor(-10051)', 'tensor(-100550)', 'tensor(-100556)', 'tensor(-100575)', 'tensor(-100637)', 'tensor(-10065)', 'tensor(-100678)', 'tensor(-10068)', 'tensor(-100693)', 'tensor(-100702)', 'tensor(-100718)', 'tensor(-100786)', 'tensor(-100798)', 'tensor(-10081)', 'tensor(-10083)', 'tensor(-100831)', 'tensor(-100834)', 'tensor(-100841)', 'tensor(-100842)', 'tensor(-100945)', 'tensor(-100956)', 'tensor(-100964)', 'tensor(-101005)', 'tensor(-101042)', 'tensor(-101044)', 'tensor(-101048)', 'tensor(-101081)', 'tensor(-101110)']
        # for node in nx_graph.nodes():
        #     if str(nx_graph.nodes[node]['node_id']) in id:
        #         start_nodes.append(node)
        # all_nodes = list(nx_graph.nodes())

        # # 随机选择N个节点
        # random.seed(66)  # 设置随机种子
        # start_nodes = random.sample(all_nodes, min(10, len(all_nodes)))
        
        print(f"Number of start nodes number: {len(start_nodes)}")
        
        return start_nodes, nx_graph
    
    def assembly_graph(self, reference_len):

        def convert_path_to_seq(path):
            # 将路径转换为序列
            dna_fragments = []
            for i in range(len(path)-1):
                node_id_cu = self.nx_graph.nodes[path[i]]['node_id'].item()
                node_id_next = self.nx_graph.nodes[path[i+1]]['node_id'].item()
                edge = self.node2edge[(str(node_id_cu),str(node_id_next))]  
                # print(f"Edge: {edge}")
                s = self.edge2seq.get(edge, "error")
                if s == "error":
                    match = re.match(r"-?(\d+\.\d+)_-?(\d+\.\d+)", edge)
                    if match:
                        num_list = sorted([str(match.group(1)), str(match.group(2))])
                        num1, num2 = num_list  # 排序后的两个数字
                        s = self.edge2seq_2[(num1, num2)]
                        seq = str(Seq(s).reverse_complement())
                        # print(f"Reverse complement: {seq}")
                else:
                    # print("SUCCESS")
                    seq= s
                if i==0:
                    dna_fragments.append(seq)
                else:
                    # print(f"Kmer: {self.kmer}")
                    dna_fragments.append(seq[self.kmer:])
            DNA_seq = ''.join(dna_fragments)
            
            return DNA_seq
        
        def path_finder_forward_stack(Graph, source_node, reference_len):
            stack = [(source_node, list(), set())]  # 栈中的元素为 (当前节点, 路径, 已访问节点集合)
            
            while stack:
                current_node, path, visited = stack.pop()
                path.append(current_node)  # 将当前节点添加到路径中
                visited.add(current_node)  # 将当前节点标记为已访问
                
                neighbors_of_node = list(Graph.successors(current_node))
                # print(f"Neighbors of node {current_node}: {neighbors_of_node}")

                next_nodes = list()
                max_score = -float('inf')  # 初始化最大分数为负无穷
                for neighbor in neighbors_of_node:
                    if neighbor not in visited:
                        # print(Graph.get_edge_data(current_node, neighbor))
                        score = Graph.get_edge_data(current_node, neighbor)[0]['score'].item()  # 获取边的分数
                        # if score > 0.3:
                        #     next_nodes.append(neighbor)
                        if score > max_score:  # 比较当前分数是否更大
                            max_score = score  # 更新最大分数
                            next_nodes = [neighbor]  # 清空并加入当前最大分数的邻居
                        elif score == max_score:  # 如果分数相等，也加入
                            next_nodes.append(neighbor)
                
                # seq = convert_path_to_seq(path)
                if not next_nodes:
                    # print(f"Final seq length: {len(seq)}, reference length: {reference_len}")  
                    sys.stdout.write(f"\r{' '*80}\r")  # 清空当前行
                    sys.stdout.write(f"Final path length: {len(path)}")
                    sys.stdout.flush()
                    yield list(path)
                else:
                    # 否则将所有邻居节点和当前路径、已访问节点压栈
                    for next_node in next_nodes:
                        stack.append((next_node, path.copy(), visited.copy()))

                # 出栈后，回退路径和访问节点集合
                path.pop()
                visited.remove(current_node)

        def greedy_path_finder(Graph, source_node, reference_len):
            path = [source_node]  # 初始化路径，从源节点开始
            visited = set([source_node])  # 初始化访问节点集合
            current_node = source_node  # 当前节点初始化为源节点
            
            while True:
                # 获取当前节点的所有邻居
                neighbors_of_node = list(Graph.successors(current_node))
                if not neighbors_of_node:
                    break  # 如果没有邻居，停止搜索

                next_node = None
                max_score = -float('inf')  # 初始化最大分数为负无穷
                max_cov = -float('inf')  # 初始化最大cov为负无穷
                candidates = []  # 存储所有最大分数的候选节点
                
                # 遍历邻居节点，选择具有最大分数的节点
                for neighbor in neighbors_of_node:
                    if neighbor not in visited:
                        score = Graph.get_edge_data(current_node, neighbor)[0]['score'].item()  # 获取边的分数
                        cov = Graph.get_edge_data(current_node, neighbor)[0]['cov'].item()
                        # 如果当前节点的score更大，更新候选节点
                        if score > max_score:
                            max_score = score
                            max_cov = cov
                            candidates = [neighbor]
                        # 如果score相等且cov更大，更新候选节点
                        elif score == max_score and cov > max_cov:
                            max_cov = cov
                            candidates = [neighbor]
                        # 如果score和cov都相等，添加到候选节点列表
                        elif score == max_score and cov == max_cov:
                            candidates.append(neighbor)
                        
                if not candidates:
                    break  # 如果没有未访问的邻居节点，结束搜索
                        
                # 随机选择一个最大分数的邻居节点
                next_node = random.choice(candidates)
                
                # 将当前选中的邻居加入路径和访问集合
                visited.add(next_node)
                path.append(next_node)
                current_node = next_node  # 更新当前节点为下一个节点
            
            # 输出最终路径长度
            sys.stdout.write(f"\r{' '*80}\r")  # 清空当前行
            sys.stdout.write(f"Final path length: {len(path)}")
            sys.stdout.flush()

            yield list(path)  # 返回路径

        print(f"Start assemblying...")
        paths = list()
        DNA_seq = list()
        Graph = self.nx_graph
        for index, node in enumerate(self.start_nodes):
            node_id = node
            if node_id is not None:
                sys.stdout.write("\n")  # 输出空行
                print(f"Node ID: {index}/{len(self.start_nodes)}: {node_id}")
                paths_generator = greedy_path_finder(Graph, node_id, reference_len)
                for path in paths_generator:
                    paths.append(path)

        print('\n')
        with open(self.contigs_filepath, 'w') as f:
            for i, path in enumerate(paths):
                print(f"Path {i+1}")
                seq = convert_path_to_seq(path)
                print(f"seq len: {len(seq)}")
                DNA_seq.append(seq)
                f.write(f">seq_{i+1}\n")  # 序列ID
                f.write(f"{seq}\n")  # 序列内容
        print(f"Number of paths: {len(DNA_seq)}")
        # print(f"Number of paths: {len(paths)}")   

if __name__ == '__main__':
    nodes_filepath = 'dataset/test/chr3.pkl'
    DGL_filepath = 'checkpoints/graph/test_0_graph.bin'
    reference_filepath = 'dataset/test/chr3.fasta'
    edges_filepath = 'dataset/test/graph.fasta'
    contigs_filepath = '/root/data1/contigs/GAT_4_contig_HiFI.fasta'
    kmer = 501

    g = AssemblyGraph(nodes_filepath, DGL_filepath, reference_filepath, edges_filepath, kmer, contigs_filepath)