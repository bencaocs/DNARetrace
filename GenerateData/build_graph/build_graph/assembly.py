import pickle
import networkx as nx
import dgl
from Bio import SeqIO
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import random

class AssemblyGraph:
    def __init__(self, nodes_filepath, DGL_filepath, reference_filepath, index, kmer, goal_len):
        self.nodes_filepath = nodes_filepath
        self.DGL_filepath = DGL_filepath
        self.goal_len = goal_len
        self.kmer = kmer
        self.start_nodes, self.end_nodes, self.reference_seq, self.reference_seq_num = self.get_start_node(reference_filepath, index, kmer)
        self.nx_graph, self.node2id = self.DGL2nx(self.DGL_filepath)
        # self.assembly_graph()
        
    def assembly_graph(self):
        def convert_path_to_seq_forward(path):
            init_seq = self.nx_graph.nodes[path[0]]['s']
            if len(path) > 1:
                for index in path[1:]:
                    init_seq += self.nx_graph.nodes[index]['s'][-1]
            return init_seq
        
        def convert_path_to_seq_backward(path):
            init_seq = self.nx_graph.nodes[path[0]]['s']
            if len(path) > 1:
                for index in path[1:]:
                    init_seq = self.nx_graph.nodes[index]['s'][0] + init_seq
            return init_seq
        
        def path_finder_forward(Graph, source_node, path = None):
            if path is None:
                path = list()

            path.append(source_node)  # 将节点添加到路径中
            # Graph.nodes[source_node]['visitied'] = True  # 将节点标记为已访问
            
            neighbors_of_node = list(Graph.successors(source_node))
            # print(f"Neighbors of node {source_node}: {neighbors_of_node}")
    
            next_nodes = list()
            for neighbor in neighbors_of_node:
                score = Graph.get_edge_data(source_node, neighbor)[0]['score'].item()  # 将Tensor转换为浮点数
                # print(f"Score of edge {self.nx_graph.nodes[source_node]['s']} -> {self.nx_graph.nodes[neighbor]['s']}: {score}")
                if score == 1.0:
                    next_nodes.append(neighbor)
            
            seq = convert_path_to_seq_forward(path)
            
            if len(next_nodes) == 0:
                # seq = convert_path_to_seq(path)
                print(f"Seq: {seq}, Length: {len(seq)}")
                yield str(seq)
            elif len(seq) == self.goal_len:
                print(f"Seq: {seq}, Length: {len(seq)}")
                yield str(seq)
            else:
                for next_node in next_nodes:
                    yield from path_finder_forward(Graph, next_node, path.copy())
                
            path.pop()
            # Graph.nodes[source_node]['visitied'] = False

        def path_finder_backward(Graph, source_node, path = None):
            if path is None:
                path = list()

            path.append(source_node)  # 将节点添加到路径中
            # Graph.nodes[source_node]['visitied'] = True  # 将节点标记为已访问
            
            neighbors_of_node = list(Graph.predecessors(source_node))
            # print(f"Neighbors of node {source_node}: {neighbors_of_node}")
    
            next_nodes = list()
            for neighbor in neighbors_of_node:
                score = Graph.get_edge_data(neighbor, source_node)[0]['score'].item()  # 将Tensor转换为浮点数
                # print(f"Score of edge {self.nx_graph.nodes[source_node]['s']} -> {self.nx_graph.nodes[neighbor]['s']}: {score}")
                if score == 1.0:
                    next_nodes.append(neighbor)
            
            seq = convert_path_to_seq_backward(path)
            
            if len(next_nodes) == 0:
                # seq = convert_path_to_seq(path)
                print(f"Seq: {seq}, Length: {len(seq)}")
                yield str(seq)
            elif len(seq) == self.goal_len:
                print(f"Seq: {seq}, Length: {len(seq)}")
                yield str(seq)
            else:
                for next_node in next_nodes:
                    yield from path_finder_backward(Graph, next_node, path.copy())
                
            path.pop()
        
        print(f"Start assemblying...")
        paths = set()
        max_consist_substring_len = list()
        Graph = self.nx_graph
        for index, node in enumerate(self.start_nodes):
            seqs = list()
            success = False
            node_id = self.node2id.get(node, None)
            if node_id is not None:
                print(f"Node ID: {index}")
                paths_generator = path_finder_forward(Graph, node_id)
                for path in paths_generator:
                    seqs.append(path)
                    if path[:self.goal_len] == self.reference_seq[index]:
                        paths.add(path[:self.goal_len])
                        max_consist_substring_len.append(self.goal_len)
                        success = True
                        break

                # seqs = sorted(seqs, key=len, reverse=True) #启动这里，114, 115, and 116 line, 124-166 line要注释掉
                # for seq in seqs:
                #     if seq == self.reference_seq[index][:len(seq)]:
                #         max_consist_substring_len.append(len(seq))
                #         break

                if not success:

                    end_node_id = self.node2id.get(self.end_nodes[index], None)
                    print(f"end_node_id:{end_node_id}")
                    seqs_backward = list()
                    if end_node_id is not None:
                        paths_generator = path_finder_backward(Graph, end_node_id)
                        for path in paths_generator:
                            seqs_backward.append(path)

                    found = False
                    for seq in seqs:
                        if len(seq) < self.goal_len:
                            len_threshold_1= self.goal_len - len(seq)
                            for seq_backward in seqs_backward:
                                if len(seq_backward) >= len_threshold_1:
                                    len_threshold_2= self.goal_len - len(seq_backward)
                                    seq_1 = seq + seq_backward[-len_threshold_1:]
                                    seq_2 = seq[:len_threshold_2] + seq_backward

                                    if seq_1 == self.reference_seq[index] or seq_2 == self.reference_seq[index]:
                                        print("Paired-end merging successful.")
                                        # print(f"Paired-end merging:{seq_1}, Length: {len(seq_1)}")
                                        # print(f"Paired-end merging:{seq_2}, Length: {len(seq_2)}")
                                        paths.add(self.reference_seq[index])
                                        max_consist_substring_len.append(self.goal_len)
                                        found = True
                                        break  # 跳出内层循环
                            if found:
                                break  # 跳出外层循环break
                    
                    if not found:
                        seqs = sorted(seqs, key=len, reverse=True)
                        for seq in seqs:
                            if seq == self.reference_seq[index][:len(seq)]:
                                max_consist_substring_len.append(len(seq))
                                break

        avg_max_consist_substring_len = np.mean(max_consist_substring_len)
        print(f"Avg_max_consist_substring_len :{avg_max_consist_substring_len}")
        print(f"Number of paths found: {len(paths)}, Seq recory rate: {(len(paths)/self.reference_seq_num)*100:.2f}%")

    def get_start_node(self, reference_filepath, index, kmer):
        sequences = [str(record.seq) for record in SeqIO.parse(reference_filepath, "fasta")]
        start_nodes = list()
        end_nodes = list()
        reference_seq = list()
        for seq in sequences:
            start_nodes.append(seq[index:index+kmer])
            end_nodes.append(seq[index+self.goal_len-kmer:index+self.goal_len])
            reference_seq.append(seq[index:index+self.goal_len])
        print(f'reference_seq_num: {len(reference_seq)}')
        
        return start_nodes, end_nodes, reference_seq, len(reference_seq)

    def DGL2nx(self, filepath):
        graphs, _ = dgl.load_graphs(filepath)
        dgl_graph = graphs[0]

        # 获取DGL图中可用的节点和边的属性
        node_attrs = list(dgl_graph.ndata.keys())  # 获取所有节点属性的名称
        edge_attrs = list(dgl_graph.edata.keys())  # 获取所有边属性的名称
        print(f"Node attributes: {node_attrs}")
        print(f"Edge attributes: {edge_attrs}")
        # 将DGL图转换为NetworkX图
        nx_graph = dgl.to_networkx(dgl_graph, edge_attrs=['score'])
        
        nodes = self.load_nodes_from_pickle(self.nodes_filepath)
        print(f"Number of nodes loaded: {len(nodes)}")
        node2id = {value[0]: key for key, value in nodes.items()}
        # 给每个节点添加属性's'
        # for node_id in nx_graph.nodes:
        #     nx_graph.nodes[node_id]['s'] = nodes[node_id][0]
            # nx_graph.nodes[node_id]['visitied'] = False 
        # for node_id, attrs in nx_graph.nodes(data=True):
        #     print(f"Node {node_id}, Attributes: {attrs}")
        
        true_edges = set()
        # 输出边的起点、终点和属性
        # for u, v, edge_attrs in nx_graph.edges(data=True):
        #     score_value = edge_attrs['score'].item()  # 将Tensor转换为浮点数
        #     # edge_id = edge_attrs.get('id', 'N/A')  # 安全获取边的ID
        #     su = nx_graph.nodes[u]['s']
        #     sv = nx_graph.nodes[v]['s']
        #     if score_value == 1.0:
        #         edge = su+sv[-1]
        #         # print(f"{su},{sv},True edge: {edge}")
        #         true_edges.add(edge)

        # print(f"Number of true edges: {len(true_edges)}")
        #################################################################
        simple_graph = nx.DiGraph(nx_graph)
        print("simple_graph type:",type(simple_graph))  
        
        seq_for_visual = self.reference_seq[:8]
        
        label_for_visual = self.reference_seq[:3]
        
        target_nodes = set()
        for seq in seq_for_visual:
            # print(len(seq))
            for i in range(0, len(seq) - self.kmer + 1):
                split = seq[i:i + self.kmer]
                node_id = node2id.get(split)
                if node_id is not None:
                    target_nodes.add(node_id)
                else:
                    pass
        print(f"Target nodes num: {len(target_nodes)}")

        target_nodes_label = list()
    
        for seq in label_for_visual:
            for i in range(0, len(seq) - self.kmer + 1):
                split = seq[i:i + self.kmer]
                node_id = node2id.get(split)
                if node_id is not None:
                    target_nodes_label.append(node_id)
                else:
                    pass
        print(f"Target label nodes num: {len(target_nodes_label)}")

        edges = dict()
        for i in range(len(target_nodes_label)-1):
            edges[(target_nodes_label[i], target_nodes_label[i+1])] = 1.0
        print(f"Edges num: {len(edges)}")

        ####################################
        ratio = 0.03

        # 获取所有边的键
        edge_keys = list(edges.keys())

        # 按比例随机选择部分键
        num_edges_to_change = int(len(edge_keys) * ratio)
        keys_to_change = random.sample(edge_keys, num_edges_to_change)

        for key in keys_to_change:
            edges[key] = 0.0
        #####################################
        subgraph = simple_graph.subgraph(target_nodes)
        print("subgraph type:",type(subgraph))

        # 绘制图形
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(subgraph, seed=1, k=0.4)  # 固定随机种子，确保每次布局一致

        # 获取边的颜色
        edge_colors = []
        for u, v, attr in subgraph.edges(data=True):
            if (u, v) in edges:
                score = edges[(u, v)]  #attr['score'].item()  # 将Tensor转换为浮点数
                edge_colors.append('#76c893' if score==1.0 else '#ff6b6b')  # 1.0 为红色，其他为黑色
            else:
                edge_colors.append((0, 0, 0, 0))
        
        node_colors = list()
        for node in subgraph.nodes():
            if node in target_nodes_label:
                if node == target_nodes_label[0]:
                    node_colors.append('#ff9f1c')
                else:
                    node_colors.append('skyblue')
            else:   
                node_colors.append('gray')
        
        # 绘制节点
        nx.draw_networkx_nodes(subgraph, pos, node_size=100, node_color=node_colors, alpha=0.8)

        # 绘制边，使用edge_colors列表设置颜色
        nx.draw_networkx_edges(subgraph, pos, edge_color=edge_colors, alpha=None)

        # 设置图形标题
        # plt.title("Visualization of Nodes 100-300", fontsize=14)
        plt.axis("off")  # 隐藏坐标轴
        # 保存图像到文件
        output_path = "graph/graph_visualization.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 300 DPI，高分辨率保存

        # 清理绘图上下文
        plt.close()

        print(f"Graph saved as {output_path}")

        ###################################################################
        # 输出每个节点的邻居节点
        # count = 0  # 初始化计数器
        # R = list()
        # for seq in self.reference_seq:
        #     edges = [seq[i:i + self.kmer + 1] for i in range(0, len(seq) - self.kmer)]
        #     end_label = len(edges)
        #     i = 0
        #     # print(f"End label: {end_label}")
        #     for edge in edges:
        #         if edge in true_edges:
        #             i += 1
        #             print(f"{i}")
        #     R.append(i/end_label)
        #     if i == end_label:
        #         count += 1
        #         print(f"True path found: {seq}")
        # mean = np.mean(R)
        # print(f"Mean: {mean}")  # 输出平均值
        # print(f"共有 {count} 条序列的所有edges在true_edges中。")

        return nx_graph, node2id
    
    def load_nodes_from_pickle(self, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

if __name__ == '__main__':
    nodes_filepath = 'dataset/test/nodes_8.pkl'
    DGL_filepath = '/home/xl/Paper/KANGNN_for_Assembly/checkpoints/graph/test_0_graph.bin'
    reference_filepath = 'dataset/test/reference_8.fasta'
    index = 0
    kmer = 15
    goal_len = 110
    g = AssemblyGraph(nodes_filepath, DGL_filepath, reference_filepath, index, kmer, goal_len)