from imports import *
import pickle

class build_debruijn_graph:
    def __init__(self, k, reads, nodes_path, edges_path, i):
        self.k = k
        self.nodes_path = nodes_path
        self.edges_path = edges_path
        self.i = i
        self.reads = self.analyze_data(reads)
        self.build_graph()
    
    def save_nodes_to_pickle(self, nodes, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(nodes, f)

    def analyze_data(self, reads):
        reads = get_reads_from_fasta(reads)
        print(f'Number of reads: {len(reads), type(reads)}')
        return reads
    
    def build_graph(self):
        nodes = dict()
        edges = set()
        deG = DeBruijnGraph(self.k)
        print('Adding seqs')
        deG.add_seqs(self.reads)
        kmers_abundance = deG.kmers # kmers is a dictionary of k-mers and their corresponding ubundance
        print(f'Number of nodes: {len(deG.kmers)}')
        print('START GENERATING NODES...')
        for index, kmer in enumerate(kmers_abundance.keys()):
            nodes[kmer] = [index, kmers_abundance[kmer]]
            
        self.save_nodes_to_pickle(nodes, self.nodes_path)
        print(f'Number of nodes: {len(nodes)}')
        
        print('START GENERATING EDGES...')
        next_bases = {'1': 'A', '2': 'T', '3': 'C', '4': 'G'}
        for kmer in nodes.keys():
            print(f'Processing k-mer {kmer}...')
            suf_sub_kmer = kmer[1:]
            
            suf_kmer1 = suf_sub_kmer + next_bases['1']
            suf_kmer2 = suf_sub_kmer + next_bases['2']
            suf_kmer3 = suf_sub_kmer + next_bases['3']
            suf_kmer4 = suf_sub_kmer + next_bases['4']
            if suf_kmer1 in nodes:
                suf_edge1 = kmer + next_bases['1']
                edges.add(suf_edge1)
            if suf_kmer2 in nodes:
                suf_edge2 = kmer + next_bases['2']
                edges.add(suf_edge2)
            if suf_kmer3 in nodes:
                suf_edge3 = kmer + next_bases['3']
                edges.add(suf_edge3)
            if suf_kmer4 in nodes:
                suf_edge4 = kmer + next_bases['4']
                edges.add(suf_edge4)
        with open(self.edges_path, "w") as fasta_file:
            for i, edge in enumerate(edges):  # 枚举每个 edge 并生成序列编号
                fasta_file.write(f">edge_{i}\n")  # 写入序列标识符（例如 edge_1, edge_2,...）
                fasta_file.write(f"{edge}\n")     # 写入实际序列

        print(f"FASTA file has been written to: {self.edges_path}")

if __name__ == '__main__':
    k = 25
    for i in range(0, 1):
    # i = 8
        reads_path = '/home/xl/Paper/KANGNN_for_Assembly/build_graph/illumina/reads/'
        nodes_path = '/home/xl/Paper/KANGNN_for_Assembly/build_graph/illumina/nodes/'
        edges_path = '/home/xl/Paper/KANGNN_for_Assembly/build_graph/illumina/edges/'
        
        fasta_file = f"{reads_path}id20_read_{i}.fasta"
        nodes_path = f"{nodes_path}nodes_{k}_{i}.pkl"
        edges_path = f"{edges_path}edges_{k}_{i}.fasta"
        
        deG = build_debruijn_graph(k, fasta_file, nodes_path, edges_path, i)