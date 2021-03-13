#
import random
import APIFu.LenSegy as LenSegy

# -------------- 初始化变量 --------------
# path_segy_add_noisy_target = 'E:/Research/data/F3_entire.segy'
path_segy_add_noisy_target = 'E:/Research/data/ArtificialData/extract_segy_valid_line_v2.segy'
path_segy_noisy_source = 'E:/Research/data/CMP_INL2000_stk_org_gain_ExtractMax.segy'


def mix_data():
    trace_add_noisy_target = LenSegy.len_read_segy_file(path_segy_add_noisy_target)['trace_raw'][:]
    # LenSegy.len_save_segy_multi_process(f3_segy_trace, path_F3_segy_file, 'F:/test_No_Mix_Noisy.segy')
    trace_noisy_source = LenSegy.len_read_segy_file(path_segy_noisy_source)['trace_raw']
    len_1_f3, len_2_f3 = len(trace_add_noisy_target), len(trace_add_noisy_target[0])
    len_1_cmp, len_2_cmp = len(trace_noisy_source), len(trace_noisy_source[0])
    random_1_mix_index = int(random.random() * len_1_cmp)
    random_2_mix_index_stable = int(random.random() * (len_2_cmp - len_2_f3))  # int()相当于带了一个-1的四舍五入加成，所以不用-1了
    print('@ 加工数据1：       ', end='')
    for i1 in range(len_1_f3):
        if random_1_mix_index >= len_1_cmp:  # 如果走完了一波，那就重新走
            random_1_mix_index = int(random.random() * len_1_cmp)
            random_2_mix_index_stable = int(random.random() * abs(len_2_cmp - len_2_f3))
        random_2_mix_index = random_2_mix_index_stable
        for i2 in range(len_2_f3):
            trace_add_noisy_target[i1][i2] += trace_noisy_source[random_1_mix_index][random_2_mix_index] * 0.7
            random_2_mix_index += 1
        random_1_mix_index += 1
        if i1 % 5 == 0:
            print('\b\b\b\b\b\b\b', '%4.1f' % (100 * i1 / len_1_f3), '%', end='')
    print('\n@ 加工结束')
    LenSegy.len_save_segy_multi_process(trace_add_noisy_target, path_segy_add_noisy_target, 'F:/test_Mix_Noisy1.segy')

    print('@ 加工数据2：       ', end='')
    random_1_mix_index2 = int(random.random() * len_1_cmp)
    random_2_mix_index_stable2 = int(random.random() * (len_2_cmp - len_2_f3))  # int()相当于带了一个-1的四舍五入加成，所以不用-1了
    for i1 in range(len_1_f3):
        if random_1_mix_index2 >= len_1_cmp:  # 如果走完了一波，那就重新走
            random_1_mix_index2 = int(random.random() * len_1_cmp)
            random_2_mix_index_stable2 = int(random.random() * abs(len_2_cmp - len_2_f3))
        random_2_mix_index = random_2_mix_index_stable2
        for i2 in range(len_2_f3):
            trace_add_noisy_target[i1][i2] += trace_noisy_source[random_1_mix_index2][random_2_mix_index] * 0.2
            random_2_mix_index += 1
        random_1_mix_index2 += 1
        if i1 % 5 == 0:
            print('\b\b\b\b\b\b\b', '%4.1f' % (100 * i1 / len_1_f3), '%', end='')
    print('\n@ 加工结束')
    LenSegy.len_save_segy_multi_process(trace_add_noisy_target, path_segy_add_noisy_target, 'F:/test_Mix_Noisy2.segy')

    print('@ 加工数据3：       ', end='')
    random_1_mix_index3 = int(random.random() * len_1_cmp)
    random_2_mix_index_stable3 = int(random.random() * (len_2_cmp - len_2_f3))  # int()相当于带了一个-1的四舍五入加成，所以不用-1了
    for i1 in range(len_1_f3):
        if random_1_mix_index3 >= len_1_cmp:  # 如果走完了一波，那就重新走
            random_1_mix_index3 = int(random.random() * len_1_cmp)
            random_2_mix_index_stable3 = int(random.random() * abs(len_2_cmp - len_2_f3))
        random_2_mix_index = random_2_mix_index_stable3
        for i2 in range(len_2_f3):
            trace_add_noisy_target[i1][i2] += trace_noisy_source[random_1_mix_index3][random_2_mix_index] * random.random()
            random_2_mix_index += 1
        random_1_mix_index3 += 1
        if i1 % 5 == 0:
            print('\b\b\b\b\b\b\b', '%4.1f' % (100 * i1 / len_1_f3), '%', end='')
    print('\n@ 加工结束')
    LenSegy.len_save_segy_multi_process(trace_add_noisy_target, path_segy_add_noisy_target, 'F:/test_Mix_Noisy3.segy')


if __name__ == '__main__':
    mix_data()
