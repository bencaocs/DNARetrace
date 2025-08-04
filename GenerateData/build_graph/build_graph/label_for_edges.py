import subprocess
from Bio import SeqIO
import os
import pickle

def blast_check_complete_match(read_fasta, reference_db, edges_path, output_file="/root/data1/blast_out/blast_results.txt"):
    """
    判断短序列是否完全匹配参考序列。
    :param short_fasta: 短序列的FASTA文件路径
    :param reference_fasta: 参考序列的BLAST数据库路径
    :param output_file: BLAST输出结果文件路径
    :return: 短序列的匹配标签（字典形式，序列ID -> 标签1或0）
    """
    # 获取短序列的长度
    short_seq = {record.id: [len(record.seq), 0] for record in SeqIO.parse(read_fasta, "fasta")}
    print(f"短序列数量: {len(short_seq)}")
    
    # 执行 BLAST 比对
    blastn_cline = [
        "blastn", 
        "-query", read_fasta, 
        "-db", reference_db, 
        "-out", output_file, 
        "-num_threads", "8",  # 使用 4 个线程
        "-outfmt", "6"  # Tabular format
    ]

    # 捕获标准输出和错误输出
    try:
        print("BLAST 比对开始...")
        result = subprocess.run(blastn_cline, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("BLAST 执行出错:")
            print(result.stderr.decode())
        else:
            print("BLAST 比对完成...")
    except Exception as e:
        print(f"BLAST 执行发生错误: {str(e)}")
        return {}

    # 解析 BLAST 结果
    print("解析 BLAST 结果...")
    if os.path.exists(output_file):  # 确保输出文件存在
        with open(output_file, "r") as f:
            for line in f:
                fields = line.strip().split("\t")
                short_seq_id = fields[0]
                alignment_length = int(fields[3])  # 比对的长度

                # 检查是否完全匹配
                if alignment_length == short_seq[short_seq_id][0]:
                    short_seq[short_seq_id][1] = 1  # 完全匹配，设置标签为 1

    with open(edges_path, 'wb') as f:
        pickle.dump(short_seq, f)


# 使用示例
reference_fasta = "build_graph/illumina/reference/id20_reference_0.fasta"
reference_db = "/home/xl/Paper/KANGNN_for_Assembly/build_graph/blast/reference_db"
read_fasta = "build_graph/illumina/edges/edges_25_0.fasta"
edges_path = 'build_graph/illumina/edges/edges_25_0.pkl'

makeblastdb_cmd = [
    "makeblastdb",
    "-in", reference_fasta,
    "-dbtype", "nucl",
    "-out", reference_db
]
subprocess.run(makeblastdb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

blast_check_complete_match(read_fasta, reference_db, edges_path)
print("匹配结束")
# for seq_id, label in result.items():
#     print(f"{seq_id}: {label[1]}")
