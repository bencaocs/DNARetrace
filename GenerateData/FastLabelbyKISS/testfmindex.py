import fm_index

# 创建 Pet 对象
fmi = fm_index.FMIndex_Uint32_KISS1()
fmi.load("/home/lzz/Projects/DBGPS_Python/seqfile.fa.fmi")
result = fmi.get_range("GCTAGCTCTAG",0)
result1 = fmi.get_offsets(result[0],result[1])
print(result1)