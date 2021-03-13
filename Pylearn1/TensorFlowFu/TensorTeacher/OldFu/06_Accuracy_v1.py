import APIFu.LenSegy as LenSegy
path_label = 'E:/Research/data/CMP_INL2000_stk_org_gain.SEGY'
path_compare = 'F:/test_result_v4.segy'

data_label = LenSegy.len_read_segy_file(path_label)['trace_raw'][:]
data_compare = LenSegy.len_read_segy_file(path_compare)['trace_raw'][:]
len_data_1v, len_data_2v = len(data_label), len(data_label[0])
result = 0
cnt = 0
for i1 in range(len_data_1v):
    for i2 in range(len_data_2v):
        if data_label[i1][i2] != 0:
            cacu = (abs(data_label[i1][i2] - data_compare[i1][i2])) / data_label[i1][i2]
        else: continue
        result += cacu
        cnt += 1
        print(i1, i2, data_label[i1][i2], data_compare[i1][i2], cacu, result)
print(result/cnt)
