import numpy as np
import struct
import sys
import os
# SA_INTV = 1
# class FMIndex:
#     def __init__(self):
#         self.OCC_INTV = 16
#         self.LOOKUP_LEN = 14
#         self.OCC1_INTV = 256
#         self.OCC2_INTV = self.OCC_INTV
#         self.B_OCC_INTV = 64

#         self.bwt_ = None
#         self.occ_ = ([], [])  # Will store lists of arrays
#         self.sa_ = None
#         self.b_ = None
#         self.b_occ_ = None
#         self.cnt_ = np.zeros(4, dtype=np.uint32)
#         self.pri_ = 0
#         self.lookup_ = []

#     def load(self, filename):
#         with open(filename, 'rb') as fin:
#             # Load cnt_
#             self.cnt_ = np.frombuffer(fin.read(4 * 4), dtype=np.uint32)
            
#             # Load pri_
#             self.pri_, = struct.unpack('I', fin.read(4))

#             # Load bwt_
#             self.bwt_ = self._load_dibit_vector(fin)

#             # Load occ_ first and second
#             occ_first_size = struct.unpack('I', fin.read(4))[0]
#             occ_first = np.frombuffer(fin.read(occ_first_size * 4 * 4), dtype=np.uint32)
#             occ_first = occ_first.reshape(occ_first_size, 4)
            
#             occ_second_size = struct.unpack('I', fin.read(4))[0]
#             occ_second = np.frombuffer(fin.read(occ_second_size * 4), dtype=np.uint8)
#             occ_second = occ_second.reshape(occ_second_size, 4)
            
#             self.occ_ = (occ_first, occ_second)

#             # Load sa_
#             self.sa_ = self._load_vector(fin, dtype=np.uint32)

#             # Load lookup_
#             lookup_size = struct.unpack('I', fin.read(4))[0]
#             self.lookup_ = np.frombuffer(fin.read(lookup_size * 4), dtype=np.uint32)

#             # Load b_ and b_occ_ if SA_INTV != 1
#             if SA_INTV != 1:
#                 self.b_ = self._load_bit_vector(fin)
#                 self.b_occ_ = self._load_vector(fin, dtype=np.uint32)

#     def _load_dibit_vector(self, fin):
#         # Implement loading for DibitVector<uint8_t>
#         size = struct.unpack('I', fin.read(4))[0]
#         return np.frombuffer(fin.read(size), dtype=np.uint8)

#     def _load_vector(self, fin, dtype):
#         size = struct.unpack('I', fin.read(4))[0]
#         return np.frombuffer(fin.read(size * np.dtype(dtype).itemsize), dtype=dtype)

#     def _load_bit_vector(self, fin):
#         # Implement loading for XbitVector<1, uint64_t>
#         size = struct.unpack('I', fin.read(4))[0]
#         return np.frombuffer(fin.read(size * 8), dtype=np.uint64)
    
# fmi = FMIndex()
# fmi.load('/home/lzz/Projects/DBGPS_Python/seqfile.fa.fmi')
os.system('./build/kISS suffix_sort seqfile.fa -k 256 -t 4 --verbose')
os.system('./build/kISS fmindex_build seqfile.fa -k 256 -t 4')
print("")