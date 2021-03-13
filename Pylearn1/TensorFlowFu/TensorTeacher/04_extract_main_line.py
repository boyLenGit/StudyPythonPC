# 去除最大值
import APIFu.LenSegy as LenSegy
import tensorflow as tf

# -------------- 初始化变量 --------------
path_segy_file = 'E:/Research/data/F3_entire.segy'
extract_up_boundary = 2800


# 提取地震数据的有效信号，也就是去噪（弱噪、强噪）
def extract_segy_valid_line():
    segy_data = LenSegy.len_read_segy_file(path_segy_file)['trace_raw'][:]
    len_1_segy, len_2_segy = len(segy_data), len(segy_data[0])

    # ------------------------------------------ 消除底噪 ------------------------------------------
    print('@ 消除底噪：       ', end='')
    for i1 in range(len_1_segy):
        for i2 in range(len_2_segy):
            if i2 <= 300:
                if abs(segy_data[i1][i2]) <= extract_up_boundary + 800:
                    segy_data[i1][i2] = 0
                continue
            if abs(segy_data[i1][i2]) <= extract_up_boundary:
                segy_data[i1][i2] = 0
        if i1 % 5 == 0:
            print('\b\b\b\b\b\b\b', '%4.1f' % (100 * i1 / len_1_segy), '%', end='')
    print('\n@ 加工结束')
    LenSegy.len_save_segy_multi_process(input_list=segy_data, read_path=path_segy_file, save_path='F:/extract_line1.segy')

    # ------------------------------------------ 消除顶噪 ------------------------------------------
    print('@ 消除顶噪：       ', end='')
    for i1 in range(len_1_segy):
        for i2 in range(len_2_segy):
            if i2 <= 65:
                segy_data[i1][i2] = 0
                continue
            else:
                break
        if i1 % 5 == 0:
            print('\b\b\b\b\b\b\b', '%4.1f' % (100 * i1 / len_1_segy), '%', end='')
    print('\n@ 加工结束')
    LenSegy.len_save_segy_multi_process(input_list=segy_data, read_path=path_segy_file, save_path='F:/extract_line2.segy')

    # ------------------------------------------ 消除强噪 ------------------------------------------
    print('@ 消除强噪：       ', end='')
    """ 添加匹配特征时，要同时在noisy_matrix_list、noisy_list_index中添加
        关于方向：匹配特征矩阵的方向=Seisee图像右转90° """
    import APIFu.Data.About_Segy as About_Segy
    noisy_matrix_list = About_Segy.noisy_features_matrix_list
    noisy_list_index = About_Segy.noisy_features_matrix_list_index
    # 过滤掉[blank, len-blank]之间的强噪
    for i2 in range(len_2_segy):
        for i1 in range(len_1_segy):
            if segy_data[i1][i2] == 0: continue
            if (len_1_segy-1-4)-i1 >= 0:
                if segy_data[i1+1][i2] != 0 and segy_data[i1+2][i2] != 0 and segy_data[i1+3][i2] != 0 and segy_data[i1+4][i2] != 0: continue
            # 遍历所有noisy_list_2v匹配特征
            for i3 in range(len(noisy_matrix_list)):
                noisy_matrix_tf = tf.constant(noisy_matrix_list[i3], dtype=tf.float32)
                i_1v, i_2v = noisy_list_index[i3][0], noisy_list_index[i3][1]
                len_matrix_1v, len_matrix_2v = len(noisy_matrix_list[i3]), len(noisy_matrix_list[i3][0])
                len_1v_sub, len_1v_add, len_2v_sub, len_2v_add = i_1v, len_matrix_1v-(i_1v+1), i_2v, len_matrix_2v-(i_2v+1)
                # 如果这个特征匹配核超限了，就换下一个匹配核
                if i1-len_1v_sub < 0 or (i1+len_1v_add > len_1_segy-1) or i2-len_2v_sub < 0 or (i2+len_2v_add > len_2_segy-1): continue
                # 没超限，则从segy_data中提取出'与匹配矩阵相同规模的矩阵'出来做运算
                matrix_temporary = []
                for i4, single_data in enumerate(segy_data[i1-len_1v_sub: i1+len_1v_add+1]):  # 遍历单个的行，+1是因为[:]提取的内容不包括后索引
                    matrix_temporary.append(single_data[i2-len_2v_sub: i2+len_2v_add+1].tolist())  # +1是因为[:]提取的内容不包括后索引
                # 复合线性滤波算法
                matrix_temporary = tf.constant(matrix_temporary, dtype=tf.float32)
                if tf.reduce_max(noisy_matrix_tf * matrix_temporary) == 0:  # 等于0代表匹配强噪特征，代表这不是有效信号，要清零
                    if tf.reduce_min(noisy_matrix_tf * matrix_temporary) == 0:
                        segy_data[i1][i2] = 0
                        break
        print('\b\b\b\b\b\b\b', '%4.1f' % (100 * i2 / len_2_segy), '%', end='')
    print('\n@ 加工结束')

    # 打印出来看看效果
    for i2 in range(len_2_segy):
        print('%5d' % i2, end='')
    print('\n')
    for i1 in range(20):
        for i2 in range(len_2_segy):
            print('%5d' % segy_data[i1][i2], end='')
        print('\n')

    LenSegy.len_save_segy_multi_process(input_list=segy_data, read_path=path_segy_file, save_path='F:/extract_line3.segy')


def remove_top_noisy():
    path_segy_file1 = 'E:/Research/data/ArtificialData/extract_segy_valid_line_v1.segy'
    segy_data = LenSegy.len_read_segy_file(path_segy_file1)['trace_raw'][:2000]
    len_1_segy, len_2_segy = len(segy_data), len(segy_data[0])
    print('@ 消除顶噪：       ', end='')
    for i1 in range(len_1_segy):
        for i2 in range(len_2_segy):
            if i2 <= 65:
                segy_data[i1][i2] = 0
                continue
            else:
                break
        if i1 % 5 == 0:
            print('\b\b\b\b\b\b\b', '%4.1f' % (100 * i1 / len_1_segy), '%', end='')
    print('\n@ 加工结束')
    LenSegy.len_save_segy_multi_process(input_list=segy_data, read_path=path_segy_file,
                                        save_path='F:/extract_line4.segy')


if __name__ == '__main__':
    extract_segy_valid_line()
    # remove_top_noisy()
