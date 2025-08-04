import pysam
from collections import defaultdict
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import random
import re

default_dict = defaultdict(set)
class_flag = set()
def read_bam(file_path):

    fasta_dict = {}
    for record in SeqIO.parse("/root/data1/illumina_data/id20.refs.fasta", "fasta"):
        fasta_dict[str(record.id)] = str(record.seq)
    
    keys = list(fasta_dict.keys())
    # print(len(keys))
    random.shuffle(keys)
    grouped_keys = [keys[i:i + 1000] for i in range(0, len(keys), 1000)]
    # print(grouped_keys)

    # 打开 BAM 文件
    # illegal_chars_all = set()
    with pysam.AlignmentFile(file_path, "rb") as bam_file:
        # 遍历比对记录
        for read in bam_file:

            # illegal_chars = re.findall(r'[^ATCG]', read.query_sequence)
            # if illegal_chars:
            #     for char in illegal_chars:
            #         illegal_chars_all.add(char)
        # print(f"发现 {len(illegal_chars_all)} 个非法碱基: {illegal_chars_all}")

        #     class_flag.add(read.flag)
        # print(class_flag)
            if read.flag==0:
                default_dict[str(bam_file.get_reference_name(read.reference_id))].add(read.query_sequence)
            # elif read.flag==2048:
            #     default_dict[int(bam_file.get_reference_name(read.reference_id))].add(read.query_sequence)
            # elif read.flag==16:
            #     seq = Seq(read.query_sequence)
            #     reverse_complement = str(seq.reverse_complement())
            #     default_dict[int(bam_file.get_reference_name(read.reference_id))].add(reverse_complement)
            # elif read.flag==2064:
            #     seq = Seq(read.query_sequence)
            #     reverse_complement = str(seq.reverse_complement())
            #     default_dict[int(bam_file.get_reference_name(read.reference_id))].add(reverse_complement)
    
    for index, key_group in enumerate(grouped_keys):
        print(f"正在处理第 {index+1} 个组...")
        ori_seq = list()
        reads_seq = list()
        for key in key_group:
            ori_seq.append([key, fasta_dict[key]])
            for id, sequence in enumerate(default_dict[key]):
                reads_seq.append([key, id, sequence])
        reads_seq = list(reads_seq)
        with open(f"id20/id20_reference_{index}.fasta", 'w') as fasta_file:
            for seq in ori_seq:
                fasta_file.write(f">{seq[0]}\n{seq[1]}\n")
        
        with open(f"id20/id20_data_{index}.fasta", 'w') as fasta_file:
            for seq2 in reads_seq:
                fasta_file.write(f">{seq2[0]}_{seq2[1]+1}\n{seq2[2]}\n")
        
    #################################################################  
        # with open("0days.fasta", 'w') as fasta_file:
        #     for key, sequences in default_dict.items():
        #         for index, sequence in enumerate(sequences):
        #             fasta_file.write(f">{key}_{index}\n{sequence}\n")
                
    # class_flag.add(read.flag)
    # print(default_dict)
        # print(class_flag)
            # if read.flag not in (0, 16):
            #     print(f'警告: 发现不正常的比对状态 {read.flag}')
            # print(f"Reference Name: {bam_file.get_reference_name(read.reference_id)}")  # 参考序列名称
            # print(f"Flag: {read.flag}")  # 比对状态标志
            # print(f"Sequence: {read.query_sequence}")
        


if __name__ == "__main__":
    bam_file_path = "id20.bam"  # 替换为你的 BAM 文件路径
    read_bam(bam_file_path)
