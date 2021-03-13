# 去除最大值
import APIFu.LenSegy as LenSegy

# -------------- 初始化变量 --------------
path_segy_file = 'E:/Research/data/CMP_INL2000_stk_org_gain.SEGY'
extract_up_boundary = 2800
extract_down_boundary = 1000


def make_data():
    segy_data = LenSegy.len_read_segy_file(path_segy_file)['trace_raw']
    len_1_segy, len_2_segy = len(segy_data), len(segy_data[0])
    print('@ 消除主线：       ', end='')
    for i1 in range(len_1_segy):
        for i2 in range(len_2_segy):
            if abs(segy_data[i1][i2]) >= extract_up_boundary:
                segy_data[i1][i2] = 0
        if i1 % 5 == 0:
            print('\b\b\b\b\b\b\b', '%4.1f' % (100 * i1 / len_1_segy), '%', end='')
    print('\n@ 加工结束')

    print('@ 数据特征增强：       ', end='')
    for i1 in range(len_1_segy):
        for i2 in range(len_2_segy):
            if abs(segy_data[i1][i2]) >= extract_down_boundary:
                segy_data[i1][i2] += segy_data[i1][i2] * 0.5
        if i1 % 5 == 0:
            print('\b\b\b\b\b\b\b', '%4.1f' % (100 * i1 / len_1_segy), '%', end='')
    print('\n@ 加工结束')
    LenSegy.len_save_segy_multi_process(input_list=segy_data, read_path=path_segy_file, save_path='F:/test_extract.segy')


if __name__ == '__main__':
    make_data()
