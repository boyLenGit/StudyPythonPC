# 实现论文内容
# ***说明***
# 改模型测试了不同网络结构下的效果，证明去掉池化、上采样的loss最小；效果排名：去掉池化上采样、1234BN > 1234BN > 0BN > 不带BN的原始网络

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, Sequential
import numpy
import segyio
import matplotlib.pyplot as plt
import random
import APIFu.LenSegy as LenSegy

len_kernel_size = 256
input_data_second_dim, input_data_third_dim = len_kernel_size, 1  # 输入网络的维度
conv_kernel_size_mnist, conv_kernel_size_ori = [28, 18, 10, 18, 28], [130, 65, 20, 65, 130]
conv_kernel_size = conv_kernel_size_ori
to_be_one = 10000.  # 地震数据归一化参数，用于将地震数据约束在-1~1之间
segy_max_value = 17024  # Max: 619101 ;average:10024.817490199499
plot_show_num = 200  # 设置Plot绘图的数据条数，与batch紧密相关。
len_origin_single_data = 0  # 原始数据单条的长度
sample_step = 64 * 3  # 将单列数据分割成多组len_kernel_size规模数据的采样步长
path_model = '/TensorFlowFu/TensorTeacher/data/001_Conv1DAE_v6.1.Len'
path_model_weights = 'D:/boyLen/Python/Pylearn1/TensorFlowFu/TensorTeacher/data/001_Conv1DAE_v6.3_45s'

# path_segy_file = 'E:/Research/data/F3_entire.segy'
path_segy_file = 'E:/Research/data/CMP_INL2000_stk_org_gain.SEGY'


def get_data_v4_segy():
    segy_data = LenSegy.len_read_segy_file(path_segy_file)['trace_raw']
    # 初始化数据
    global len_origin_single_data
    len_origin_single_data = len(segy_data[0])
    # ↓ segy的列数、行数调整，以适应网络模型（单条数据的长度必须是4的倍数，因为有两次2维池化和2维上采样）
    x_test_all_data = segy_data[:]  # 复制原数据
    # 【Lite】原始数据
    x_test_all_noisy = len_make_noisy(x_test_all_data.copy(), '原始数据')
    # 数据类型转化与缩放
    x_test_all_data, x_test_all_noisy = (x_test_all_data/to_be_one).astype(numpy.float32), (x_test_all_noisy/to_be_one).astype(numpy.float32)
    # 维度变形
    x_test_all_data, x_test_all_noisy = tf.expand_dims(x_test_all_data, axis=2), tf.expand_dims(x_test_all_noisy, axis=2)
    # Batch化。x_test_all_data为全部的原始数据，x_test_all_noisy为全部加了噪声的原始数据
    x_test_all_dataset = tf.data.Dataset.from_tensor_slices((x_test_all_data, x_test_all_noisy)).batch(batch_size=plot_show_num)
    return x_test_all_dataset


def len_make_noisy(input_list, func_name=''):
    print('@ {0}-生成带噪声数据：       '.format(func_name), end='')
    len_input = len(input_list)
    for i1 in range(len_input):
        for i2 in range(len(input_list[0])):
            input_list[i1][i2] += (random.random() * segy_max_value * 0.25)
        if i1 % 20 == 0: print('\b\b\b\b\b\b\b', '%4.1f' % (100*i1/len_input), '%',  end='')
    print('')
    return input_list  # 返回已经加噪声的数据


def boylen_kernel_generator_for_train(input_list):  # 【核采样】指定宽度采样法（仅对x_train进行处理即可）
    enlarge_data_list = []
    len_input_list = len(input_list)
    len_input_list_single = len(input_list[0])
    print('boyLen核采样(train集)：       ', end='')
    for i1 in range(len_input_list):
        one_trace_cut_times = int(10 * random.random())
        step_add_all = 0
        for i2 in range(one_trace_cut_times):
            step_single = int(random.random() * (len_input_list_single-len_kernel_size-1)/3)
            step_add_all += step_single
            if (step_add_all+len_kernel_size-1) >= len(input_list[0]): break
            enlarge_data_list.append(input_list[i1][step_add_all: step_add_all+len_kernel_size])
        if i1 % 20 == 0:
            print('\b\b\b\b\b\b\b', '%4.1f' % (100*i1/len_input_list), '%',  end='')
    print('')
    return enlarge_data_list


class One_Conv(keras.Model):
    def __init__(self):
        super(One_Conv, self).__init__()
        self.sq1 = Sequential([
            layers.Conv1D(filters=40, kernel_size=conv_kernel_size[0], activation='relu', padding='same'),
            layers.BatchNormalization(),
            # layers.MaxPool1D(pool_size=2),
            layers.Conv1D(filters=40, kernel_size=conv_kernel_size[1], activation='relu', padding='same'),
            layers.BatchNormalization(),
            # layers.MaxPool1D(pool_size=2),
            layers.Conv1D(filters=40, kernel_size=conv_kernel_size[2], activation='relu', padding='same'),
            layers.BatchNormalization(),
            # layers.UpSampling1D(size=2),
            layers.Conv1D(filters=40, kernel_size=conv_kernel_size[3], activation='relu', padding='same'),
            layers.BatchNormalization(),
            # layers.UpSampling1D(size=2),
            layers.Conv1D(filters=input_data_third_dim, kernel_size=conv_kernel_size[4], activation='tanh', padding='same')
        ])

    def call(self, inputs, training=True, mask=None):
        out1 = self.sq1(inputs)
        # self.sq1.summary()
        return out1


def boylen_train(db_all):
    # 初始化数据
    global len_origin_single_data
    # ------------------------------------- 初始化并训练模型 ---------------------------------------
    model1 = One_Conv()
    model1.build(input_shape=(None, None, input_data_third_dim))
    model1.load_weights(path_model_weights)
    # ------------------------------------- 生成训练完预测出来的segy数据 ---------------------------------------
    list_result_all_fusion, list_noisy_all_fusion, list_input_all_fusion = [], [], []
    len_data_all = len(db_all)
    print('@ 训练完毕，开始生成全规模数据：     ', end='')
    for step, (x_all, x_all_noisy) in enumerate(db_all):
        x_all_result_single = model1(x_all_noisy)
        list_result_all_fusion.extend(x_all_result_single)
        list_noisy_all_fusion.extend(x_all_noisy)
        list_input_all_fusion.extend(x_all)
        if step % 5 == 0:
            print('\b\b\b\b\b\b\b', '%4.1f' % (100 * step / len_data_all), '%', end='')
    print('')
    # ------------------------------------- 预测数据单条全部融合完毕，画个图看看效果 ---------------------------------------
    # 维度改变
    list_result_all_fusion, list_noisy_all_fusion, list_input_all_fusion = \
        tf.reshape(list_result_all_fusion, [-1, len_origin_single_data]), \
        tf.reshape(list_noisy_all_fusion, [-1, len_origin_single_data]), \
        tf.reshape(list_input_all_fusion, [-1, len_origin_single_data])
    # 数据转化，从(-1,1)扩张到原始大小
    list_result_all_fusion, list_noisy_all_fusion, list_input_all_fusion = \
        (list_result_all_fusion.numpy()) * to_be_one, \
        (list_noisy_all_fusion.numpy()) * to_be_one, \
        (list_input_all_fusion.numpy()) * to_be_one
    # 绘图，看效果用的
    boylen_plot(numpy.array(list_input_all_fusion), title='Original All Fusion')
    boylen_plot(numpy.array(list_noisy_all_fusion), title='Noisy All Fusion')
    boylen_plot(numpy.array(list_result_all_fusion), title='Result All Fusion')
    # 多进程同步保存Segy文件
    LenSegy.len_save_segy_multi_process(list_noisy_all_fusion, path_segy_file, 'J:/test_noisy.segy')
    LenSegy.len_save_segy_multi_process(list_result_all_fusion, path_segy_file, 'F:/test_result.segy')


def boylen_plot(input_list, line_num=plot_show_num, title='title'):
    list_x = []
    plt.figure(figsize=(12, 12))
    for i0 in range(len(input_list[0])):  # ← 生成x轴的标签数据
        list_x.append(i0)
    for i1 in range(line_num):
        for i2 in range(len(input_list[0])):
            input_list[i1][i2] += 30000 * i1
    for i3 in range(line_num):
        upper, lower = 3000 + i3 * 30000, -3000 + i3 * 30000
        big_list = numpy.ma.masked_where(input_list[i3] <= upper - 500, input_list[i3])
        mid_list = numpy.ma.masked_where((input_list[i3] <= lower - 3000) | (input_list[i3] >= upper + 3000), input_list[i3])
        lit_list = numpy.ma.masked_where(input_list[i3] >= lower + 500, input_list[i3])
        plt.plot(big_list, list_x, 'b-', mid_list, list_x, 'r-', lit_list, list_x, 'b-', linewidth=2)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    db_all = get_data_v4_segy()
    boylen_train(db_all=db_all)
