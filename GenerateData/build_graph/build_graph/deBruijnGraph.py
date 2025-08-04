from imports import *

class DeBruijnGraph:
    def __init__(self, kmer_len=21):
        self.kmers = {}
        self.kmer_len = kmer_len

    def add_seqs(self, seqs):
        for seq in seqs:
            self.add_seq(seq)

    def add_seq(self, str):
        if len(str) >= self.kmer_len:
            i = 0
            kmstr = ''
            while i <= len(str) - self.kmer_len:
                kmstr = str[i:i + self.kmer_len]
                if not ("N" in kmstr):
                    self.add_kmer(kmstr)
                    # self.add_kmer(DNA_rev_complement(kmstr))
                else:
                    if kmstr.count("N") == 1:
                        self.add_kmer(kmstr.replace("N", "A"))
                        self.add_kmer(kmstr.replace("N", "T"))
                        self.add_kmer(kmstr.replace("N", "G"))
                        self.add_kmer(kmstr.replace("N", "C"))
                        # self.add_kmer(DNA_rev_complement(kmstr.replace("N", "A")))
                        # self.add_kmer(DNA_rev_complement(kmstr.replace("N", "T")))
                        # self.add_kmer(DNA_rev_complement(kmstr.replace("N", "G")))
                        # self.add_kmer(DNA_rev_complement(kmstr.replace("N", "C")))
                i = i + 1

    def add_kmer(self, kmer, num=1):
        # comp_kmer = self.compressDNA(kmer)
        if kmer in self.kmers:
            self.kmers[kmer] = self.kmers[kmer] + num
        else:
            self.kmers[kmer] = num