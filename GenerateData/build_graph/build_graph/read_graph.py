# import os
# import glob
# import dgl

# # 定义保存图的路径
# save_train_graph_path = "/home/xl/Paper/KANGNN_for_Assembly/build_graph/illumina/graph"
# train_bin_files = glob.glob(os.path.join(save_train_graph_path, "*.bin"))
# # 遍历每个 .bin 文件并加载图
# for file_path in train_bin_files:
#     graph_list, _ = dgl.load_graphs(file_path)
#     train_graph = graph_list[0]
#     num_nodes = train_graph.num_nodes()
#     num_edges = train_graph.num_edges()
#     print("Graph {} has {} nodes and {} edges.".format(file_path, num_nodes, num_edges))
reference_seq = "AGCCTGGTCCGGGGTTGATTTCCCCTCGTAATACGTCCGTCATACAGGGACCGAGCACGGTGGGGATTATAGTCCTAGACCCACGGCTGAGTGTATGATGGCCTTTGTTC"
seqs = [
    "AGCCTGGTCCGGGGTTGATTTCCCCTCGTAATACGTCCGTCATACAGGGACCGAGCACGGTGGGGATTATAGTCCTAGACCCACGGCTGAGTGTATGATGGCCTTTGTTC"
                                                "TGGACCGAGCACGGTGGGGATTATAGTCCTAGACCCACGGCTGAGTGTATGATGGCCTTTGTTC",
    "TGGTGGGGATTATAGTCCTAGACCCACGGCTGAGTGTATGATGGCCTTTGTTC",
    "CAGTCCTAGACCCACGGCTGAGTGTATGATGGCCTTTGTTC",
    "GGTGTATGATGGCCTTTGTTC"
]

print(len(reference_seq))
# 检查每条序列是否是 reference_seq 的子序列
for i, seq in enumerate(seqs):
    if seq in reference_seq:
        print(f"Seq {i+1} is a part of reference_seq.")
    else:
        print(f"Seq {i+1} is NOT a part of reference_seq.")
