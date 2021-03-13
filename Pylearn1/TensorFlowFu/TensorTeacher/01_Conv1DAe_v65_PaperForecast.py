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
import APIFu.LenDataProcess as LenDataProcess

len_kernel_size = 256
input_data_second_dim, input_data_third_dim = len_kernel_size, 1  # 输入网络的维度
conv_kernel_size_mnist, conv_kernel_size_ori = [28, 18, 10, 18, 28], [130, 65, 20, 65, 130]
conv_kernel_size = conv_kernel_size_ori
to_be_one = 10000.  # 地震数据归一化参数，用于将地震数据约束在-1~1之间
segy_max_value = 17024  # Max: 619101 ;average:10024.817490199499
plot_show_num = 200  # 设置Plot绘图的数据条数，与batch紧密相关。
len_origin_single_data = 0  # 原始数据单条的长度
sample_step = 64 * 3  # 将单列数据分割成多组len_kernel_size规模数据的采样步长
path_model_weights_load = 'D:/boyLen/Python/Pylearn1/TensorFlowFu/TensorTeacher/data/01_Conv1DAE/v63_final'

# path_segy_file = 'E:/Research/data/F3_entire.segy'
path_segy_file = 'E:/Research/data/CMP_INL2000_stk_org_gain.SEGY'


def get_data_v4_segy():
    segy_data = LenSegy.len_read_segy_file(path_segy_file)['trace_raw']
    # 初始化数据
    global len_origin_single_data
    len_origin_single_data = len(segy_data[0])
    x_test_all_dataset = LenDataProcess.segy_to_tf_for_result_1output(data=segy_data, to_be_one=to_be_one, reshape_dim=None, expand_axis=2, batch_size=plot_show_num)
    return x_test_all_dataset


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


def boylen_forecast(db_all):
    # 初始化数据
    global len_origin_single_data
    # ------------------------------------- 初始化并训练模型 ---------------------------------------
    model1 = One_Conv()
    model1.build(input_shape=(None, None, input_data_third_dim))
    model1.load_weights(path_model_weights_load)
    # ------------------------------------- 生成训练完预测出来的segy数据 ---------------------------------------
    list_result_all_fusion = []
    len_data_all = len(db_all)
    print('@ 训练完毕，开始生成全规模数据：     ', end='')
    for step, x_all in enumerate(db_all):
        x_all_result_single = model1(x_all)
        list_result_all_fusion.extend(x_all_result_single)
        if step % 5 == 0:
            print('\b\b\b\b\b\b\b', '%4.1f' % (100 * step / len_data_all), '%', end='')
    print('')
    # ------------------------------------- 预测数据单条全部融合完毕，画个图看看效果 ---------------------------------------
    # 维度改变
    list_result_all_fusion = tf.reshape(list_result_all_fusion, [-1, len_origin_single_data])
    # 数据转化，从(-1,1)扩张到原始大小
    list_result_all_fusion = (list_result_all_fusion.numpy()) * to_be_one
    # 绘图，看效果用的
    boylen_segy_plot(numpy.array(list_result_all_fusion), title='Result All Fusion')
    LenSegy.len_save_segy_multi_process(list_result_all_fusion, path_segy_file, 'F:/Conv1DAE_result_v1.segy')
    print(tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation()))


def boylen_segy_plot(input_list, line_num=plot_show_num, title='title'):
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
    boylen_forecast(db_all=db_all)
